"""Model backends over plain HTTP (no SDKs).

Every backend offers two things:
  choose(system, user, choices, image) -> {"category", "reason"}   (schema-constrained)
  embed(texts) -> list of vectors

The category is constrained to an enum at decode time where the runtime
supports it, so a small model cannot invent a folder name.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request

from .config import ModelSettings


class BackendError(RuntimeError):
    pass


def _post(url: str, body: dict, headers: dict | None = None, timeout: float = 180) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        raise BackendError(f"{e.code} from {url}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise BackendError(f"cannot reach {url}: {getattr(e, 'reason', e)}") from None


def _get(url: str, timeout: float = 5) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        raise BackendError(f"cannot reach {url}: {getattr(e, 'reason', e)}") from None


def choice_schema(choices: list[str]) -> dict:
    # "reason" comes first so the model writes a sentence before committing.
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "maxLength": 200},
            "category": {"type": "string", "enum": choices},
        },
        "required": ["reason", "category"],
        "additionalProperties": False,
    }


def parse_choice(text: str, choices: list[str]) -> dict:
    """Defensive parse for runtimes that ignore the schema."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {}
    cat = str(data.get("category", "")).strip()
    if cat not in choices:
        # tolerate case or slash differences, e.g. "finance / taxes"
        norm = {re.sub(r"\W", "", c.lower()): c for c in choices}
        cat = norm.get(re.sub(r"\W", "", cat.lower()), "")
    return {"category": cat or None, "reason": str(data.get("reason", ""))[:300]}


def _mime(image: bytes) -> str:
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    if image[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


class Backend:
    def __init__(self, settings: ModelSettings, embed_model: str | None = None):
        self.s = settings
        self.embed_model = embed_model

    def choose(self, system: str, user: str, choices: list[str], image: bytes | None = None) -> dict:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def status(self) -> dict:
        """{'ok': bool, 'models': [...], 'detail': str}"""
        raise NotImplementedError


class Ollama(Backend):
    def choose(self, system, user, choices, image=None):
        msg = {"role": "user", "content": user}
        if image:
            msg["images"] = [base64.b64encode(image).decode()]
        body = {
            "model": self.s.name,
            "messages": [{"role": "system", "content": system}, msg],
            "format": choice_schema(choices),
            "stream": False,
            "options": {"temperature": self.s.temperature, "num_ctx": 4096},
            "keep_alive": "10m",
        }
        if _maybe_thinking(self.s.name):
            body["think"] = False  # classification needs no chain of thought; 3-10x faster
        r = _post(f"{self.s.url}/api/chat", body, timeout=self.s.timeout)
        return parse_choice(r["message"]["content"], choices)

    def embed(self, texts):
        r = _post(f"{self.s.url}/api/embed", {"model": self.embed_model, "input": texts, "keep_alive": "10m"},
                  timeout=self.s.timeout)
        return r["embeddings"]

    def status(self):
        try:
            tags = _get(f"{self.s.url}/api/tags")
        except BackendError as e:
            return {"ok": False, "models": [], "detail": str(e)}
        return {"ok": True, "models": [m["name"] for m in tags.get("models", [])], "detail": "Ollama is running"}


class OpenAICompatible(Backend):
    """LM Studio, llama.cpp server, vLLM, Jan, LocalAI, mlx-lm server, OpenRouter, ..."""

    def _headers(self):
        key = self.s.api_key or os.environ.get("OPENAI_API_KEY")
        return {"Authorization": f"Bearer {key}"} if key else {}

    def choose(self, system, user, choices, image=None):
        content: str | list = user
        if image:
            uri = f"data:{_mime(image)};base64,{base64.b64encode(image).decode()}"
            content = [{"type": "text", "text": user}, {"type": "image_url", "image_url": {"url": uri}}]
        body = {
            "model": self.s.name,
            "temperature": self.s.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "classification", "strict": True,
                                                "schema": choice_schema(choices)}},
        }
        r = _post(f"{self.s.url.rstrip('/')}/chat/completions", body, self._headers(), self.s.timeout)
        return parse_choice(r["choices"][0]["message"]["content"] or "", choices)

    def embed(self, texts):
        r = _post(f"{self.s.url.rstrip('/')}/embeddings", {"model": self.embed_model, "input": texts},
                  self._headers(), self.s.timeout)
        return [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]

    def status(self):
        try:
            req = urllib.request.Request(f"{self.s.url.rstrip('/')}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
        except Exception as e:
            return {"ok": False, "models": [], "detail": f"cannot reach {self.s.url}: {e}"}
        return {"ok": True, "models": [m["id"] for m in data.get("data", [])], "detail": "server is running"}


class Anthropic(Backend):
    """Cloud, opt-in. Uses forced tool use to get a schema-shaped answer."""

    def _key(self):
        key = self.s.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise BackendError("set ANTHROPIC_API_KEY to use the anthropic backend")
        return key

    def choose(self, system, user, choices, image=None):
        content: list = [{"type": "text", "text": user}]
        if image:
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": _mime(image),
                                                            "data": base64.b64encode(image).decode()}})
        body = {
            "model": self.s.name,
            "max_tokens": 400,
            "temperature": self.s.temperature,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "tools": [{"name": "file_it", "description": "File the document into one category.",
                       "input_schema": choice_schema(choices)}],
            "tool_choice": {"type": "tool", "name": "file_it"},
        }
        r = _post(f"{self.s.url.rstrip('/')}/v1/messages", body,
                  {"x-api-key": self._key(), "anthropic-version": "2023-06-01"}, self.s.timeout)
        block = next((b for b in r.get("content", []) if b.get("type") == "tool_use"), {})
        return parse_choice(json.dumps(block.get("input", {})), choices)

    def embed(self, texts):
        raise BackendError("the anthropic backend has no embeddings; use strategy = \"llm\"")

    def status(self):
        try:
            self._key()
        except BackendError as e:
            return {"ok": False, "models": [], "detail": str(e)}
        return {"ok": True, "models": [self.s.name], "detail": "API key found"}


def _maybe_thinking(model: str) -> bool:
    return any(t in model.lower() for t in ("qwen3", "deepseek-r1", "gpt-oss", "magistral", "phi4-reasoning"))


def make_backend(settings: ModelSettings, embed_model: str | None = None) -> Backend:
    kinds = {"ollama": Ollama, "openai": OpenAICompatible, "anthropic": Anthropic}
    if settings.backend not in kinds:
        raise BackendError(f"unknown backend {settings.backend!r}; choose one of {', '.join(kinds)}")
    return kinds[settings.backend](settings, embed_model)
