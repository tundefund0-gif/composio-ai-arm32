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
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: Any = None,
        retryable: bool = True,
        retry_after: Optional[float] = None,
    ):
        self.status_code = status_code
        self.body = body
        self.retryable = retryable
        self.retry_after = retry_after
        super().__init__(message)


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
        # Fallback model support
        self.fallback_model = config.opencode_fallback_model or None
        self.fallback_base_url = config.opencode_fallback_base_url or None
        self.fallback_api_key = config.opencode_fallback_api_key or None

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.7,
             stream: bool = False, tools: Optional[List[Dict]] = None,
             retries: int = 2) -> Union[LLMResponse, Generator[str, None, None]]:
        """Send a chat request with automatic retry and optional model fallback on failure."""
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

        import random as _random
        last_err = None
        for attempt in range(retries + 1):
            try:
                return self._sync(body)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_err = e
                logger.warning("LLM call attempt %d/%d failed: %s", attempt + 1, retries + 1, e)
                if attempt < retries:
                    delay = 1.0 * (2 ** attempt) + _random.uniform(0, 0.5)
                    logger.info("Retrying in %.1fs...", delay)
                    time.sleep(delay)
            except LLMError as e:
                last_err = e
                if not e.retryable:
                    logger.warning("Non-retryable error: %s", e)
                    if self.fallback_model:
                        logger.info("Trying fallback model: %s", self.fallback_model)
                        return self._fallback_chat(messages, temperature, tools, retries)
                    raise
                logger.warning("LLM call attempt %d/%d failed: %s", attempt + 1, retries + 1, e)
                if attempt < retries:
                    delay = e.retry_after if e.retry_after is not None else 1.0 * (2 ** attempt) + _random.uniform(0, 0.5)
                    logger.info("Retrying in %.1fs...", delay)
                    time.sleep(delay)

        # Try fallback after all retries exhausted
        if self.fallback_model:
            logger.info("All retries exhausted (%d), trying fallback model: %s", retries + 1, self.fallback_model)
            return self._fallback_chat(messages, temperature, tools, retries)
        raise LLMError(f"LLM API failed after {retries + 1} attempts: {last_err}")

    def _fallback_chat(self, messages, temperature, tools, retries):
        """Execute chat using fallback model configuration."""
        original_model = self.model
        original_url = self.base_url
        original_key = self.api_key
        try:
            self.model = self.fallback_model
            if self.fallback_base_url:
                self.base_url = self.fallback_base_url.rstrip("/")
            if self.fallback_api_key:
                self.api_key = self.fallback_api_key
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            if tools:
                body["tools"] = tools
            logger.info("Falling back to model=%s at %s", self.model, self.base_url)
            return self._sync(body)
        finally:
            self.model = original_model
            self.base_url = original_url
            self.api_key = original_key

    def _sync(self, body: Dict) -> LLMResponse:
        with httpx.Client(timeout=config.llm_timeout) as cl:
            r = cl.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers())
            if r.status_code >= 400:
                raise self._http_error(r)
            return LLMResponse(r.json())

    def _stream(self, body: Dict) -> Generator[str, None, None]:
        """Stream tokens. Also captures tool calls and emits __tool_calls__:... when detected."""
        with httpx.Client(timeout=config.llm_timeout) as cl:
            with cl.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=self._headers()) as r:
                if r.status_code >= 400:
                    r.read()
                    raise self._http_error(r)

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
        if not self.api_key:
            raise LLMError("OPENGATE_API_KEY is required", retryable=False)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _http_error(self, resp: httpx.Response) -> LLMError:
        try:
            body: Any = resp.json()
        except Exception:
            body = resp.text

        retry_after = self._retry_after(resp.headers.get("retry-after"))
        retryable = self._is_retryable(resp.status_code, body, retry_after)
        err_type, err_msg = self._error_details(body)

        detail = f"LLM API error: HTTP {resp.status_code}"
        if err_type:
            detail += f" {err_type}"
        if err_msg:
            detail += f" - {err_msg}"
        elif body:
            detail += f" - {body}"

        return LLMError(
            detail,
            status_code=resp.status_code,
            body=body,
            retryable=retryable,
            retry_after=retry_after,
        )

    @staticmethod
    def _retry_after(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            return max(float(value), 0.0)
        except ValueError:
            return None

    @staticmethod
    def _error_details(body: Any) -> tuple[Optional[str], Optional[str]]:
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return err.get("type"), err.get("message")
            if isinstance(err, str):
                return None, err
        return None, None

    def _is_retryable(self, status_code: int, body: Any, retry_after: Optional[float]) -> bool:
        err_type, _ = self._error_details(body)
        if err_type == "FreeUsageLimitError":
            return False
        if status_code == 429:
            return retry_after is not None
        return status_code == 408 or status_code >= 500
