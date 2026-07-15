"""Zen Agent — orchestrates LLM + Composio tool ecosystem with long-text support."""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from config import config
from core.composio_client import ComposioClient, ComposioAPIError
from core.llm_client import LLMClient, LLMResponse, LLMError

logger = logging.getLogger("zen-agent")



def _parse_dsml_tool_calls(text):
    """Parse DSML/XML formatted tool calls from model text output.
    Handles formats like:
      <tool_calls><invoke name="TOOL">...</invoke></tool_calls>
      <||DSML||tool_calls><||DSML||invoke name="TOOL">...</||DSML||invoke></||DSML||tool_calls>
      <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
    """
    if text:
        text = text.replace(chr(0xFF5C), chr(0x007C))
    import re as _re
    if not text:
        return None
    
    calls = []
    
    # --- Format 1: <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call> ---
    # Match <tool_call>...</tool_call> blocks
    tool_call_blocks = _re.findall(
        _re.compile(r"<tool_call[^>]*>(.*?)</tool_call[^>]*>", _re.DOTALL),
        text
    )
    for block in tool_call_blocks:
        # Extract function name from <function=NAME> or <function>NAME</function>
        fn_match = _re.search(r"<function[=]?(.*?)>", block)
        if not fn_match:
            continue
        fn_name = fn_match.group(1).strip()
        # Extract parameters from <parameter=KEY>VALUE</parameter> or <parameter name="KEY">VALUE</parameter>
        params = _re.findall(
            _re.compile(r'<parameter(?:\s+name\s*=\s*"?|=)\s*([a-zA-Z_]\w*)"?\s*[^>]*>(.*?)</parameter[^>]*>', _re.DOTALL),
            block
        )
        args = {}
        for pname, pval in params:
            pval = pval.strip()
            try:
                args[pname] = json.loads(pval)
            except Exception:
                args[pname] = pval
        if fn_name:
            calls.append({
                "id": "call_" + str(abs(hash(fn_name)) % 10**6),
                "type": "function",
                "function": {"name": fn_name, "arguments": json.dumps(args)},
            })
    
    # --- Format 2: DSML/XML <[||]tool_calls> with <[||]invoke> tags ---
    if not calls:
        # Pattern: optional ||DSML|| prefix before tag names
        _pd = r"(?:[|]{2}DSML[|]{2})?"
        outer = _re.search(_pd + r"<tool_calls[^>]*>(.*?)</" + _pd + r"tool_calls[^>]*>", text, _re.DOTALL)
        inner = outer.group(1) if outer else text
        
        # Parse individual <invoke> tags
        invokes = _re.findall(
            _pd + r'<invoke\s+name="([a-zA-Z_]\w*)"[^>]*>(.*?)</' + _pd + r'invoke[^>]*>',
            inner, _re.DOTALL
        )
        if invokes:
            for name, body in invokes:
                params = _re.findall(
                    _pd + r'<parameter\s+(?:name|value)\s*=\s*"([a-zA-Z_]\w*)"[^>]*>(.*?)</' + _pd + r'parameter[^>]*>',
                    body, _re.DOTALL
                )
                args = {}
                for pname, pval in params:
                    pval = pval.strip()
                    try:
                        args[pname] = json.loads(pval)
                    except Exception:
                        args[pname] = pval
                calls.append({
                    "id": "call_" + str(abs(hash(name)) % 10**6),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })
    
    return calls if calls else None


def _strip_dsml_tags(text):
    """Remove DSML/XML markup from text, keeping only natural language content."""
    if text:
        text = text.replace(chr(0xFF5C), chr(0x007C))
    import re as _re
    # Remove <tool_call>...</tool_call> blocks
    text = _re.sub(_re.compile(r"<tool_call[^>]*>.*?</tool_call[^>]*>", _re.DOTALL), "", text)
    # Remove <function[=...]>...</function> blocks
    text = _re.sub(_re.compile(r"<function[^>]*>.*?</function[^>]*>", _re.DOTALL), "", text)
    # Remove <parameter...>...</parameter> blocks (handles both <parameter=KEY> and <parameter name="KEY"> formats)
    text = _re.sub(_re.compile(r'<parameter(?:\s+name\s*=\s*"?|=)\s*[a-zA-Z_]\w*"?\s*[^>]*>.*?</parameter[^>]*>', _re.DOTALL), "", text)
    # Match tags with optional pipe decoration: <||DSML||tool_calls||>, <||tool_calls>, <tool_calls>
    _pipes = "[" + chr(124) + chr(0xFF5C) + "]*"
    _dsml = "(?:" + _pipes + "DSML" + _pipes + ")?"
    _tag = lambda name: "<" + _pipes + _dsml + name + _pipes + "[^>]*>.*?</" + _pipes + _dsml + name + _pipes + "[^>]*>"
    text = _re.sub(_re.compile(_tag("tool_calls"), _re.DOTALL), "", text)
    text = _re.sub(_re.compile(_tag("invoke"), _re.DOTALL), "", text)
    # Also clean up any standalone <function=NAME> tags without closing </function>
    text = _re.sub(_re.compile(r'<function\s*=.*?>', _re.DOTALL), "", text)
    return text.strip()


def _normalize_text(text):
    """Clean up common formatting issues: fix missing spaces, normalize whitespace."""
    if not text:
        return text
    import re as _re
    # Fix missing space after period that ends a sentence (word.word -> word. word)
    # But NOT URLs (connect.composio.dev) or dots in the middle of abbreviations
    text = _re.sub(r'(?<=[a-z])\.(?=[A-Z][a-z])', '. ', text)
    # Fix missing space after comma
    text = _re.sub(r',([a-zA-Z])', r', \1', text)
    # Collapse multiple spaces
    text = _re.sub(r' +', ' ', text)
    return text.strip()


META_TOOL_NAMES = {
    "COMPOSIO_SEARCH_TOOLS",
    "COMPOSIO_GET_TOOL_SCHEMAS",
    "COMPOSIO_MANAGE_CONNECTIONS",
    "COMPOSIO_REMOTE_WORKBENCH",
    "COMPOSIO_REMOTE_BASH_TOOL",
    "COMPOSIO_MULTI_EXECUTE_TOOL",
}


def _normalize_tool_calls(calls):
    """Normalize parsed tool calls: wrap non-meta tools into COMPOSIO_MULTI_EXECUTE_TOOL."""
    if not calls:
        return calls
    normalized = []
    for tc in calls:
        fn_name = tc["function"]["name"]
        if fn_name in META_TOOL_NAMES:
            normalized.append(tc)
        else:
            # Model output a raw Composio tool slug (e.g. GITHUB_LIST_REPOSITORIES...)
            # Route it through COMPOSIO_MULTI_EXECUTE_TOOL
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            # If args already has "tools" array, use it; otherwise wrap the tool
            if "tools" in args and isinstance(args["tools"], list):
                tools_payload = args["tools"]
            else:
                tools_payload = [{"tool_slug": fn_name, "arguments": args}]
            normalized.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": "COMPOSIO_MULTI_EXECUTE_TOOL",
                    "arguments": json.dumps({"tools": tools_payload}),
                },
            })
            logger.info("Routing non-meta tool %s -> COMPOSIO_MULTI_EXECUTE_TOOL", fn_name)
    return normalized


META_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_SEARCH_TOOLS",
            "description": "Search for Composio tools relevant to a task. Returns tool slugs, descriptions, and schemas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "use_case": {"type": "string", "description": "NL description of what the user wants"}
                            },
                            "required": ["use_case"],
                        },
                    }
                },
                "required": ["queries"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_GET_TOOL_SCHEMAS",
            "description": "Get input/output schemas for specific tool slugs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_slugs": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tool_slugs"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_MANAGE_CONNECTIONS",
            "description": "Check or create OAuth connections for toolkits. Returns auth links if user needs to connect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "toolkits": {"type": "array", "items": {"type": "string"}},
                    "reinitiate_all": {"type": "boolean", "description": "Force reconnection even if active"},
                },
                "required": ["toolkits"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_REMOTE_WORKBENCH",
            "description": "Execute Python code in remote sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_to_execute": {"type": "string", "description": "Python code to execute"},
                    "language": {"type": "string", "enum": ["python"]},
                },
                "required": ["code_to_execute"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_REMOTE_BASH_TOOL",
            "description": "Run shell commands in remote sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["cmd"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "COMPOSIO_MULTI_EXECUTE_TOOL",
            "description": "Execute multiple tools in parallel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_slug": {"type": "string"},
                                "arguments": {"type": "object"},
                            },
                            "required": ["tool_slug"],
                        },
                    }
                },
                "required": ["tools"],
            },
        }
    },
]


def _truncate_msgs(messages: List[Dict], max_count: int = 60) -> List[Dict]:
    """Keep system prompt + last N messages to stay within context limits."""
    if len(messages) <= max_count:
        return messages
    # Keep system (first) + last max_count-1
    return [messages[0]] + messages[-(max_count - 1):]


class ZenAgent:
    """AI agent with 1,000+ Composio tools, long-text, streaming & tool loops."""

    def __init__(self, user_id: str, session_id: Optional[str] = None,
                 toolkits: Optional[List[str]] = None, enable_sandbox: bool = True):
        self.user_id = user_id
        self.session_id = session_id
        self.toolkits = toolkits
        self.enable_sandbox = enable_sandbox
        self._composio = ComposioClient()
        self._llm = LLMClient()
        self._messages: List[Dict] = []
        self._init_session()

    def _init_session(self):
        if self.session_id:
            try:
                self._composio.get_session(self.session_id)
                logger.info("Reusing session %s", self.session_id)
                return
            except ComposioAPIError:
                logger.info("Session %s not found, creating new", self.session_id)
                self.session_id = None
        s = self._composio.create_session(
            user_id=self.user_id, toolkits=self.toolkits, sandbox=self.enable_sandbox
        )
        self.session_id = s["session_id"]
        logger.info("Created session %s", self.session_id)

    def _sysprompt(self) -> str:
        return f"""You are **Zen Agent** — an AI assistant with access to **23,790+ tools** across 1,000+ apps via Composio.

## 🎯 Core Rules

**You can ONLY call the 6 meta tools listed below.**
You CANNOT call individual app tools directly (like GMAIL_*, GITHUB_*, SLACK_*, etc.).
Those must be executed through COMPOSIO_MULTI_EXECUTE_TOOL.

## 🔧 Available Functions

1. **COMPOSIO_SEARCH_TOOLS** — Search for tools by describing what the user wants
2. **COMPOSIO_GET_TOOL_SCHEMAS** — Get detailed input/output schemas for specific tool slugs
3. **COMPOSIO_MANAGE_CONNECTIONS** — Check or create OAuth connections (returns auth links if needed)
4. **COMPOSIO_REMOTE_WORKBENCH** — Execute Python code in a remote sandbox
5. **COMPOSIO_REMOTE_BASH_TOOL** — Run shell commands in a remote sandbox
6. **COMPOSIO_MULTI_EXECUTE_TOOL** — Execute multiple tools in parallel (use this to call any app tool)

## 📋 Workflow

1. **Search** — Use COMPOSIO_SEARCH_TOOLS to find relevant tool slugs for the user's request
2. **Check connections** — Use COMPOSIO_MANAGE_CONNECTIONS to verify OAuth status
3. **Execute** — Use COMPOSIO_MULTI_EXECUTE_TOOL with tool slugs and their arguments
4. **Code** — For data tasks, use COMPOSIO_REMOTE_WORKBENCH (Python sandbox)
5. **Iterate** — If more info is needed, search and execute again with new context

## ⚠️ Required Parameters

- **COMPOSIO_MANAGE_CONNECTIONS**: needs `"toolkits"` array like `["github", "gmail"]`
- **COMPOSIO_SEARCH_TOOLS**: needs `"queries"` array with `"use_case"` strings
- **COMPOSIO_MULTI_EXECUTE_TOOL**: needs `"tools"` array with `"tool_slug"` and `"arguments"`
- **COMPOSIO_REMOTE_WORKBENCH**: needs `"code_to_execute"` with Python code
- **COMPOSIO_REMOTE_BASH_TOOL**: needs `"cmd"` with shell command
- **COMPOSIO_GET_TOOL_SCHEMAS**: needs `"tool_slugs"` array of slug strings

## 📌 Example

COMPOSIO_MULTI_EXECUTE_TOOL with tools array:
- tool_slug: "GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER"
- arguments: {{"per_page": 5}}

## 🚫 What NOT to Do

- NEVER invent function names — only use the 6 listed above
- NEVER use XML, DSML, or markup like <tool_calls>, <invoke>, ||DSML||, <function=...>
- ALWAYS use the API's structured function_call format (native tool calls)
- **Stop calling tools once you have enough info to answer the question**
- Do NOT call the same tool repeatedly with the same parameters
- If a tool fails, check the error, fix parameters, and retry
- For multi-step tasks, plan all steps upfront then execute efficiently

## ✨ Formatting Guidelines

- Use **bold** for emphasis and `code` for technical terms
- Present lists, tables, and results with clear formatting
- Be concise, accurate, and helpful
- When showing code or results, use proper markdown code blocks
- After executing tools, summarize the results clearly for the user

---

*Session: {self.session_id} | User: {self.user_id}*
*Current UTC: {datetime.now(timezone.utc).isoformat()}*
""" 
    # ── Public API ──────────────────────────────────────────
    def chat(self, message: str, stream: bool = False) -> LLMResponse | Generator[str, None, None]:
        self._messages.append({"role": "user", "content": message})
        if stream:
            return self._stream()
        return self._sync()

    def get_history(self) -> List[Dict]:
        return list(self._messages)

    def clear_history(self):
        self._messages = []

    def get_info(self) -> Dict:
        usage = self.total_token_usage()
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "sandbox_enabled": self.enable_sandbox,
            "toolkits": self.toolkits,
            "model": self._llm.model,
            "message_count": len(self._messages),
            "total_input_chars": usage["input_chars"],
            "total_output_chars": usage["output_chars"],
            "total_tool_calls": usage["tool_calls"],
        }

    # ── Sync path (REST) ─────────────────────────────────────
    def _sync(self) -> LLMResponse:
        msgs = _truncate_msgs(self._build_msgs(), config.max_history_messages)
        resp = self._llm.chat(msgs, tools=META_TOOL_DEFS, retries=2)
        if resp.tool_calls:
            return self._handle_tools(resp, msgs)
        # Model may output tool calls as DSML text instead of proper tool_calls
        dsml_calls = _parse_dsml_tool_calls(resp.content) if resp.content else None
        if dsml_calls:
            logger.info("DSML: found %d tool calls in text, executing", len(dsml_calls))
            resp.tool_calls = _normalize_tool_calls(dsml_calls)
            resp.message["content"] = _strip_dsml_tags(resp.content) or None
            return self._handle_tools(resp, msgs)
        # Strip any DSML/XML markup from response text
        if resp.content:
            cleaned = _strip_dsml_tags(resp.content)
            if cleaned != resp.content:
                resp.message["content"] = cleaned or None
                resp.content = cleaned or ""
        # Normalize text formatting
        if resp.content:
            resp.content = _normalize_text(resp.content)
            resp.message["content"] = resp.content
        self._messages.append({"role": "assistant", "content": resp.content})
        return resp

    def _handle_tools(self, resp: LLMResponse, msgs: List[Dict], _depth: int = 0) -> LLMResponse:
        if _depth > 8:
            logger.warning("Tool call depth exceeded (limit 8), generating summary")
            # Do a final non-streaming call asking for summary
            summary_msgs = [msgs[0], {"role": "user", "content": "Summarize what was accomplished so far based on the tool results. Be concise."}]
            tool_count = 0
            for m in reversed(msgs):
                if m.get("role") == "tool" and tool_count < 3:
                    summary_msgs.insert(1, m)
                    tool_count += 1
            try:
                final = self._llm.chat(summary_msgs, retries=1)
                if final and final.content:
                    cleaned = _normalize_text(_strip_dsml_tags(final.content))
                    final.message["content"] = cleaned
                    final.content = cleaned
                    final.tool_calls = None
                else:
                    final = LLMResponse({"choices": [{"message": {"content": "The tools have been executed. The results are shown above."}}], "model": self._llm.model})
            except Exception:
                final = LLMResponse({"choices": [{"message": {"content": "I completed the requested operations. The results have been processed."}}], "model": self._llm.model})
            self._messages.append({"role": "assistant", "content": final.content or ""})
            return final
        # Build assistant message for context — strip DSML tool_calls, use clean format
        assistant_entry = {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {"id": tc["id"], "type": "function", "function": tc["function"]}
                for tc in (resp.tool_calls or [])
            ] or None,
        }
        msgs.append(assistant_entry)
        self._messages.append(assistant_entry)
        # Execute each tool call
        tool_results_text = []
        for tc in resp.tool_calls or []:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            result = self._exec_composio(fn, args)
            result_str = json.dumps(result, default=str)[:min(config.max_tool_results_length, 8000)]
            logger.info("Tool %s result (depth=%d): %d chars, has_error=%s", fn, _depth, len(result_str), "error" in result)
            tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result_str}
            msgs.append(tool_msg)
            self._messages.append(tool_msg)
            tool_results_text.append(result_str)
        # Call LLM with tool results in context
        msgs = _truncate_msgs(msgs, config.max_history_messages)
        logger.info("Follow-up LLM call: depth=%d, msgs=%d", _depth, len(msgs))
        final = None
        try:
            final = self._llm.chat(msgs, retries=2)
        except LLMError as e:
            logger.warning("Follow-up LLM call failed (depth=%d, attempt 1): %s", _depth, e)
            # Retry with simplified context: system + user + last tool results only
            try:
                simplified = [msgs[0]]  # system prompt
                for m in msgs:
                    if m.get("role") == "user":
                        simplified.append(m)
                        break
                for m in msgs:
                    if m.get("role") == "tool":
                        simplified.append(m)
                logger.info("Retry with simplified context: %d msgs", len(simplified))
                final = self._llm.chat(simplified, retries=1)
            except LLMError as e2:
                logger.warning("Simplified retry also failed (depth=%d): %s", _depth, e2)
        if final is None:
            # Build a readable fallback from actual tool results
            fallback_parts = []
            for tr in tool_results_text:
                try:
                    parsed = json.loads(tr)
                    err = None
                    if isinstance(parsed, dict):
                        err = parsed.get("error") or parsed.get("details")
                        if not err and "data" in parsed and isinstance(parsed["data"], dict):
                            err = parsed["data"].get("error")
                    if err:
                        fallback_parts.append("⚠️ " + str(err)[:500])
                    elif isinstance(parsed, dict) and parsed.get("data"):
                        result_data = parsed["data"]
                        if isinstance(result_data, dict) and result_data:
                            fallback_parts.append(json.dumps(result_data, indent=2)[:3000])
                        else:
                            fallback_parts.append("Tool executed successfully but returned no data.")
                    else:
                        fallback_parts.append(json.dumps(parsed, indent=2)[:3000] if isinstance(parsed, dict) else str(tr)[:3000])
                except Exception:
                    fallback_parts.append(str(tr)[:3000])
            fallback_content = _normalize_text("\n\n".join(fallback_parts) if fallback_parts else "Tool executed, but the AI could not generate a summary.")
            final = LLMResponse({
                "choices": [{"message": {"content": fallback_content}, "finish_reason": "stop"}],
                "model": self._llm.model,
            })
        # Model may output tool calls as DSML text in the follow-up response
        dsml_calls = _parse_dsml_tool_calls(final.content) if final.content else None
        if dsml_calls:
            logger.info("DSML: found %d tool calls in follow-up text (depth=%d)", len(dsml_calls), _depth)
            final.tool_calls = _normalize_tool_calls(dsml_calls)
            final.message["content"] = _strip_dsml_tags(final.content) or None
            final.content = final.message["content"] or ""
            return self._handle_tools(final, msgs, _depth + 1)
        # Clean any remaining DSML/XML markup
        if final.content:
            cleaned = _strip_dsml_tags(final.content)
            if cleaned != final.content:
                final.message["content"] = cleaned or None
                final.content = cleaned or ""
        # Normalize text formatting
        if final.content:
            final.content = _normalize_text(final.content)
            final.message["content"] = final.content
        self._messages.append({"role": "assistant", "content": final.content})
        return final

    # ── Streaming path (WebSocket) ───────────────────────────
    def _stream(self) -> Generator[str, None, None]:
        """Stream with full tool call loop support."""
        msgs = _truncate_msgs(self._build_msgs(), config.max_history_messages)
        gen = self._llm.chat(msgs, stream=True, tools=META_TOOL_DEFS, retries=2)
        if not isinstance(gen, Generator):
            return

        depth = 0
        while True:
            full_content = ""
            tool_calls_data = None

            # Buffer content tokens — don't yield yet, DSML tags span multiple tokens
            content_buffer = []
            for token in gen:
                if token.startswith("__reasoning__"):
                    yield token
                elif token.startswith("__tool_calls__:"):
                    try:
                        tool_calls_data = json.loads(token[len("__tool_calls__:"):])
                        tc_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls_data]
                        logger.info("Stream: received __tool_calls__ with %d calls: %s", len(tool_calls_data), tc_names)
                    except json.JSONDecodeError:
                        tool_calls_data = None
                else:
                    full_content += token
                    content_buffer.append(token)

            # Check for DSML tool calls in buffered text
            dsml_found = False
            if not tool_calls_data and full_content:
                dsml_calls = _parse_dsml_tool_calls(full_content)
                if dsml_calls:
                    dsml_found = True
                    tool_names = [c["function"]["name"] for c in dsml_calls]
                    logger.info("DSML: found %d tool calls in streamed text (depth=%d): %s", len(dsml_calls), depth, tool_names)
                    logger.debug("DSML raw text (first 300): %s", full_content[:300])
                    dsml_calls = _normalize_tool_calls(dsml_calls)
                    tool_calls_data = [
                        {"id": c["id"], "function": c["function"]}
                        for c in dsml_calls
                    ]
                    # Yield only the non-DSML prefix text if any, then status
                    cleaned = _strip_dsml_tags(full_content)
                    cleaned = _normalize_text(cleaned)
                    if cleaned:
                        yield cleaned + "\n\n"
                    yield "__status__:⚙️ Executing tools (step " + str(depth + 1) + ")..."

            # If no DSML and no tool calls, yield buffered content now
            if not dsml_found and not tool_calls_data:
                cleaned = _strip_dsml_tags(full_content) or full_content
                cleaned = _normalize_text(cleaned)
                if cleaned:
                    yield cleaned
                # Still save even if empty
                self._messages.append({"role": "assistant", "content": cleaned or "(no response)"})
                break

            # If no DSML but we have proper tool_calls from API, yield content now
            if not dsml_found and tool_calls_data:
                for t in content_buffer:
                    cleaned_t = _strip_dsml_tags(t)
                    if cleaned_t:
                        yield cleaned_t

            # Execute tools and loop
            assistant_msg = {
                "role": "assistant",
                "content": full_content.strip() or None,
                "tool_calls": [],
            }
            for tc in tool_calls_data:
                tc_id = tc.get("id", f"call_{abs(hash(str(tc))) % 10**6}")
                entry = {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                assistant_msg["tool_calls"].append(entry)

            msgs.append(assistant_msg)
            self._messages.append(assistant_msg)

            for tc in tool_calls_data:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                result = self._exec_composio(fn, args)
                result_str = json.dumps(result, default=str)[:config.max_tool_results_length]
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{abs(hash(str(tc))) % 10**6}"),
                    "content": result_str,
                }
                msgs.append(tool_msg)
                self._messages.append(tool_msg)

            depth += 1
            if depth > 8:
                logger.warning("Streaming tool call depth exceeded (limit 8)")
                # Use tool results for a final summary instead of breaking with error
                summary_msgs = [msgs[0], {"role": "user", "content": "Summarize what was accomplished so far based on the tool results. Be concise."}]
                # Add last few tool results for context
                tool_count = 0
                for m in reversed(msgs):
                    if m.get("role") == "tool" and tool_count < 3:
                        summary_msgs.insert(1, m)
                        tool_count += 1
                try:
                    final_resp = self._llm.chat(summary_msgs, retries=1)
                    final_text = final_resp.content or ""
                    final_text = _normalize_text(_strip_dsml_tags(final_text))
                    yield final_text
                except Exception:
                    yield "__status__:⚠️ Complex task completed. Results shown above."
                break

            gen = self._llm.chat(msgs, stream=True, retries=1)
            if not isinstance(gen, Generator):
                break

    # ── Session persistence ──────────────────────────────────
    @property
    def _history_dir(self) -> Path:
        p = Path(config.data_dir) / "conversations"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_session(self, path: Optional[str] = None) -> str:
        """Save conversation history to a JSON file. Returns the file path."""
        if path is None:
            path = str(self._history_dir / f"{self.user_id}_{uuid.uuid4().hex[:12]}.json")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "toolkits": self.toolkits,
            "sandbox_enabled": self.enable_sandbox,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "messages": self._messages,
        }
        p.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Session saved to %s (%d messages)", p, len(self._messages))
        return str(p)

    def load_session(self, path: str) -> None:
        """Load conversation history from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        data = json.loads(p.read_text())
        ver = data.get("version", 1)
        self._messages = data.get("messages", [])
        if ver < 2:
            self._messages = [m for m in self._messages if m.get("role") != "system"]
        logger.info("Session loaded from %s (%d messages)", p, len(self._messages))

    def export_markdown(self, path: Optional[str] = None) -> str:
        """Export conversation as Markdown. Returns the file path."""
        if path is None:
            path = str(self._history_dir / f"{self.user_id}_{uuid.uuid4().hex[:12]}.md")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        parts = [f"# Zen Agent - Conversation", f"User: {self.user_id}"]
        for msg in self._messages:
            role = msg.get("role", "unknown")
            text = msg.get("content", "") or ""
            if role == "user":
                parts.append("\n## 👤 You\n\n" + text + "\n")
            elif role == "assistant":
                tc = msg.get("tool_calls")
                if tc:
                    fn_list = ", ".join(t.get("function", {}).get("name", "?") for t in tc)
                    parts.append("\n## 🤖 AI (calling " + fn_list + ")\n\n" + text + "\n")
                else:
                    parts.append("\n## 🤖 AI\n\n" + text + "\n")
            elif role == "tool":
                snippet = text[:200] + "..." if len(text) > 200 else text
                parts.append("\n**Tool result:** `" + snippet + "`\n")
        p.write_text("".join(parts))
        return str(p)

    def total_token_usage(self) -> Dict[str, int]:
        """Calculate cumulative token usage from all LLM responses stored in messages."""
        total_input = 0
        total_output = 0
        total_tool_calls = 0
        for msg in self._messages:
            if msg.get("role") == "assistant":
                tc = msg.get("tool_calls")
                if tc:
                    total_tool_calls += len(tc)
        # Estimate: count chars as rough proxy when we don't have actual token counts
        for msg in self._messages:
            content = msg.get("content", "") or ""
            if msg.get("role") == "user":
                total_input += len(content)
            elif msg.get("role") == "assistant":
                total_output += len(content)
        return {
            "input_chars": total_input,
            "output_chars": total_output,
            "tool_calls": total_tool_calls,
            "message_count": len(self._messages),
        }

    # ── Tool execution ───────────────────────────────────────
    def _exec_composio(self, action: str, args: Dict) -> Dict[str, Any]:
        import concurrent.futures
        try:
            # Run tool execution with 30s timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(self._composio.execute_meta, self.session_id, action, args)
                return fut.result(timeout=30)
        except concurrent.futures.TimeoutError:
            logger.warning("Tool %s timed out after 30s", action)
            return {"error": f"Tool execution timed out after 30 seconds", "action": action}
        except ComposioAPIError as e:
            logger.error("Tool %s failed: %s", action, e)
            err_body = str(e.body)[:500] if e.body else ""
            return {
                "error": str(e),
                "details": err_body,
                "available_functions": ["COMPOSIO_SEARCH_TOOLS", "COMPOSIO_GET_TOOL_SCHEMAS",
                    "COMPOSIO_MANAGE_CONNECTIONS", "COMPOSIO_REMOTE_WORKBENCH",
                    "COMPOSIO_REMOTE_BASH_TOOL", "COMPOSIO_MULTI_EXECUTE_TOOL"],
                "hint": f"Tool '{action}' failed. If this is an app tool (like GMAIL_*), "
                        f"you must execute it via COMPOSIO_MULTI_EXECUTE_TOOL with tool_slug='{action}'."
            }

    def _build_msgs(self) -> List[Dict]:
        return [{"role": "system", "content": self._sysprompt()}, *self._messages]
