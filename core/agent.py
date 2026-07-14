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
        outer = _re.search(
            _re.compile(
                "<" + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?"
                "tool_calls[^>]*>(.*?)</"
                "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?tool_calls[^>]*>",
                _re.DOTALL
            ),
            text
        )
        inner = outer.group(1) if outer else text
        
        # Parse individual <invoke> tags
        tag_re = _re.compile(
            "<" + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?"
            "invoke" + chr(32) + "+name=" + chr(34) + "([a-zA-Z_]\\w*)" + chr(34)
            + "[^>]*>(.*?)</"
            + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?invoke[^>]*>",
            _re.DOTALL
        )
        invokes = tag_re.findall(inner)
        if invokes:
            param_re = _re.compile(
                "<" + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?"
                "parameter" + chr(32) + "+name=" + chr(34) + "([a-zA-Z_]\\w*)" + chr(34)
                + "[^>]*>(.*?)</"
                + "(?:[" + chr(124) + "][" + chr(124) + "]DSML[" + chr(124) + "][" + chr(124) + "])?parameter[^>]*>",
                _re.DOTALL
            )
            for name, body in invokes:
                params = param_re.findall(body)
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
        return f"""You are Zen Agent, an AI assistant with access to 23,790+ tools via Composio.

**CRITICAL - You can ONLY call the 6 meta tools listed below.**
You CANNOT call individual app tools directly (like GMAIL_*, GITHUB_*, SLACK_*, etc.).
Those must be executed through COMPOSIO_MULTI_EXECUTE_TOOL.

**Available Functions (only these 6):**
1. COMPOSIO_SEARCH_TOOLS - Search for tools by use case description
2. COMPOSIO_GET_TOOL_SCHEMAS - Get input/output schemas for specific tool slugs
3. COMPOSIO_MANAGE_CONNECTIONS - Check or create OAuth connections for toolkits
4. COMPOSIO_REMOTE_WORKBENCH - Execute Python code in remote sandbox
5. COMPOSIO_REMOTE_BASH_TOOL - Run shell commands in remote sandbox
6. COMPOSIO_MULTI_EXECUTE_TOOL - Execute multiple tools in parallel

**Workflow:**
1. Use COMPOSIO_SEARCH_TOOLS to find relevant tools for the user's request.
2. Check connections with COMPOSIO_MANAGE_CONNECTIONS. If not active, show the user the auth link.
3. Execute tools via COMPOSIO_MULTI_EXECUTE_TOOL (pass tool_slug and arguments).
4. For complex tasks, use COMPOSIO_REMOTE_WORKBENCH.

**IMPORTANT — Required Parameters:**
- COMPOSIO_MANAGE_CONNECTIONS always requires "toolkits" (array of toolkit names like ["github", "gmail"]).
- COMPOSIO_SEARCH_TOOLS always requires "queries" (array of objects with "use_case").
- COMPOSIO_MULTI_EXECUTE_TOOL always requires "tools" (array of objects with "tool_slug").

**Example of executing a GitHub tool:**
COMPOSIO_MULTI_EXECUTE_TOOL with tools=[{{"tool_slug": "GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER", "arguments": {{"per_page": 2}}}}]

**NEVER invent function names that are not listed above.**
Only call the 6 functions listed above. If you need a tool not listed, use COMPOSIO_SEARCH_TOOLS first.

**CRITICAL FORMATTING RULES:**
- Use ONLY the native function calling protocol. Do NOT write tool calls as XML, DSML, or any markup format.
- Do NOT output tags like <tool_calls>, <invoke>, <parameter>, ||DSML||, or similar markup.
- If you output tool calls as text/XML/DSML instead of using function calling, the tools will NOT execute properly.
- Always use the structured function_call format provided by the API.

Session: {self.session_id} | User: {self.user_id}
Current UTC time: {datetime.now(timezone.utc).isoformat()}
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
        self._messages.append({"role": "assistant", "content": resp.content})
        return resp

    def _handle_tools(self, resp: LLMResponse, msgs: List[Dict], _depth: int = 0) -> LLMResponse:
        if _depth > 5:
            logger.warning("Tool call depth exceeded, stopping")
            final = resp
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
            fallback_content = "\n\n".join(fallback_parts) if fallback_parts else "Tool executed but the AI model could not generate a summary."
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
                    logger.info("DSML: found %d tool calls in streamed text (depth=%d)", len(dsml_calls), depth)
                    dsml_calls = _normalize_tool_calls(dsml_calls)
                    tool_calls_data = [
                        {"id": c["id"], "function": c["function"]}
                        for c in dsml_calls
                    ]
                    # Yield only the non-DSML prefix text if any, then status
                    cleaned = _strip_dsml_tags(full_content)
                    if cleaned:
                        yield cleaned + "\n\n"
                    yield "⚙️ Executing tool calls...\n"

            # If no DSML and no tool calls, yield buffered content now
            if not dsml_found and not tool_calls_data:
                for t in content_buffer:
                    yield _strip_dsml_tags(t) or t
                self._messages.append({"role": "assistant", "content": _strip_dsml_tags(full_content) or full_content})
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
            if depth > 5:
                logger.warning("Streaming tool call depth exceeded")
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
        try:
            return self._composio.execute_meta(self.session_id, action, args)
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
