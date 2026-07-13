"""OpenAI-compatible LLM client — retry, long text, reasoning, streaming."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional, Union

import httpx

from config import config

logger = logging.getLogger("llm-client")


class LLMError(Exception):
    pass


class LLMResponse:
    """Wraps a chat completion response."""
    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw
        self.choice = raw.get("choices", [{}])[0]
        self.message = self.choice.get("message", {})
        self.content: str = self.message.get("content", "") or ""
        self.reasoning: str = self.message.get("reasoning_content", "") or ""
        self.finish_reason: str = self.choice.get("finish_reason", "")
        self.model: str = raw.get("model", "")
        self.usage: Dict = raw.get("usage", {})
        self.tool_calls = self.message.get("tool_calls")
        self.input_tokens: int = self.usage.get("prompt_tokens", 0) if self.usage else 0
        self.output_tokens: int = self.usage.get("completion_tokens", 0) if self.usage else 0

    def __repr__(self) -> str:
        t = len(self.tool_calls) if self.tool_calls else 0
        return f"LLMResponse(model={self.model}, in={self.input_tokens}, out={self.output_tokens}, tools={t})"

    def to_dict(self) -> Dict:
        return {"content": self.content, "reasoning": self.reasoning[:500],
                "model": self.model, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


class LLMClient:
    """Client for OpenAI-compatible chat APIs with retry & long text support."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, max_tokens: Optional[int] = None):
        self.api_key = api_key or config.opencode_api_key
        self.base_url = (base_url or config.opencode_base_url).rstrip("/")
        self.model = model or config.opencode_model
        self.max_tokens = max_tokens or config.opencode_max_tokens

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.7,
             stream: bool = False, tools: Optional[List[Dict]] = None,
             retries: int = 2) -> Union[LLMResponse, Generator[str, None, None]]:
        """Send a chat request with automatic retry on failure."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools

        if stream:
            return self._stream(body)

        last_err = None
        for attempt in range(retries + 1):
            try:
                return self._sync(body)
            except (httpx.TimeoutException, httpx.ConnectError, LLMError) as e:
                last_err = e
                logger.warning("LLM call attempt %d/%d failed: %s", attempt + 1, retries + 1, e)
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM API failed after {retries + 1} attempts: {last_err}")

    def _sync(self, body: Dict) -> LLMResponse:
        with httpx.Client(timeout=config.llm_timeout) as cl:
            r = cl.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers())
            if r.status_code >= 400:
                try:
                    d = r.json()
                except Exception:
                    d = r.text
                raise LLMError(f"LLM API error: HTTP {r.status_code} - {d}")
            return LLMResponse(r.json())

    def _stream(self, body: Dict) -> Generator[str, None, None]:
        """Stream tokens. Also captures tool calls and emits __tool_calls__:... when detected."""
        with httpx.Client(timeout=config.llm_timeout) as cl:
            with cl.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=self._headers()) as r:
                if r.status_code >= 400:
                    try:
                        d = r.json()
                    except Exception:
                        d = r.text
                    raise LLMError(f"LLM API error: HTTP {r.status_code} - {d}")

                tool_call_deltas: Dict[int, Dict] = {}
                finish_reason = None

                for line in r.iter_lines():
                    if not line or line.startswith(":keep-alive"):
                        continue
                    if line.startswith("data: "):
                        d = line[6:].strip()
                        if d == "[DONE]":
                            break
                        try:
                            chunk = json.loads(d)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            fr = choices[0].get("finish_reason")
                            if fr:
                                finish_reason = fr

                            rc = delta.get("reasoning_content", "")
                            if rc:
                                yield f"__reasoning__{rc}"

                            c = delta.get("content", "")
                            if c:
                                yield c

                            tcs = delta.get("tool_calls", [])
                            for tc in tcs:
                                idx = tc.get("index", 0)
                                if idx not in tool_call_deltas:
                                    tool_call_deltas[idx] = {
                                        "id": tc.get("id", ""),
                                        "type": tc.get("type", "function"),
                                        "function": {"name": "", "arguments": ""},
                                    }
                                cur = tool_call_deltas[idx]
                                tc_fn = tc.get("function", {})
                                if tc.get("id"):
                                    cur["id"] = tc["id"]
                                if tc_fn.get("name"):
                                    cur["function"]["name"] += tc_fn["name"]
                                if tc_fn.get("arguments"):
                                    cur["function"]["arguments"] += tc_fn["arguments"]

                        except json.JSONDecodeError:
                            continue

                if tool_call_deltas and finish_reason == "tool_calls":
                    sorted_calls = [tool_call_deltas[i] for i in sorted(tool_call_deltas)]
                    yield f"__tool_calls__:{json.dumps(sorted_calls)}"

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages).content

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
