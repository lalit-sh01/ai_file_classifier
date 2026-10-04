"""Model backends over plain HTTP (no SDKs), all on this computer or your local network.

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

from .config import ModelSettings, check_offline


class BackendError(RuntimeError):
    pass


class ModelTimeout(BackendError):
    """The server is up but one answer took too long (e.g. a large image on a CPU-only machine)."""


def _post(url: str, body: dict, headers: dict | None = None, timeout: float = 180) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        raise BackendError(f"{e.code} from {url}: {detail}") from None
    except TimeoutError:
        raise ModelTimeout(f"no answer from {url} within {timeout:.0f}s") from None
    except (urllib.error.URLError, ConnectionError) as e:
        if isinstance(getattr(e, "reason", None), TimeoutError):
            raise ModelTimeout(f"no answer from {url} within {timeout:.0f}s") from None
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


def _loads(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


def parse_choice(text: str, choices: list[str]) -> dict:
    """Defensive parse for runtimes that ignore the schema."""
    data = _loads(text)
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

    def choose(self, system: str, user: str, choices: list[str], image: bytes | None = None,
               model: str | None = None) -> dict:
        raise NotImplementedError

    def generate(self, system: str, user: str, schema: dict, model: str | None = None) -> dict:
        """Free-form structured output (used by `fclass discover`)."""
        raise NotImplementedError

    def warm(self, model: str) -> None:
        """Load a model before timing-sensitive calls. Loading from disk can take minutes on a cold machine."""

    def has_model(self, name: str) -> bool:
        st = self.status()
        return st["ok"] and any(m == name or m == f"{name}:latest" or m.split(":")[0] == name for m in st["models"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def status(self) -> dict:
        """{'ok': bool, 'models': [...], 'detail': str}"""
        raise NotImplementedError


class Ollama(Backend):
    LOAD_TIMEOUT = 900  # a 3 GB model read from a slow disk took over 3 minutes on a test machine

    def warm(self, model: str) -> None:
        # An empty prompt loads the model and returns; keep_alive keeps it ready between files.
        _post(f"{self.s.url}/api/generate", {"model": model, "prompt": "", "keep_alive": "10m"},
              timeout=self.LOAD_TIMEOUT)

    def _chat(self, system, user, schema, image=None, model=None, max_tokens=300) -> str:
        model = model or self.s.name
        msg = {"role": "user", "content": user}
        if image:
            msg["images"] = [base64.b64encode(image).decode()]
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, msg],
            "format": schema,
            "stream": False,
            # A cap on output: small models can ramble inside a JSON string until a timeout.
            "options": {"temperature": self.s.temperature, "num_ctx": 4096, "num_predict": max_tokens},
            "keep_alive": "10m",
        }
        if _maybe_thinking(model):
            body["think"] = False  # classification needs no chain of thought; 3-10x faster
        timeout = max(self.s.timeout, 300) if image else self.s.timeout  # pictures are slower to read
        r = _post(f"{self.s.url}/api/chat", body, timeout=timeout)
        return r["message"]["content"]

    def choose(self, system, user, choices, image=None, model=None):
        return parse_choice(self._chat(system, user, choice_schema(choices), image, model), choices)

    def generate(self, system, user, schema, model=None):
        return _loads(self._chat(system, user, schema, model=model, max_tokens=400))

    def embed(self, texts):
        r = _post(f"{self.s.url}/api/embed", {"model": self.embed_model, "input": texts, "keep_alive": "10m"},
                  timeout=max(self.s.timeout, self.LOAD_TIMEOUT) if not getattr(self, "_embed_ready", False)
                  else self.s.timeout)
        self._embed_ready = True
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

    def _chat(self, system, user, schema, image=None, model=None, max_tokens=300) -> str:
        content: str | list = user
        if image:
            uri = f"data:{_mime(image)};base64,{base64.b64encode(image).decode()}"
            content = [{"type": "text", "text": user}, {"type": "image_url", "image_url": {"url": uri}}]
        body = {
            "model": model or self.s.name,
            "temperature": self.s.temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        }
        timeout = max(self.s.timeout, 300) if image else self.s.timeout
        r = _post(f"{self.s.url.rstrip('/')}/chat/completions", body, self._headers(), timeout)
        return r["choices"][0]["message"]["content"] or ""

    def choose(self, system, user, choices, image=None, model=None):
        return parse_choice(self._chat(system, user, choice_schema(choices), image, model), choices)

    def generate(self, system, user, schema, model=None):
        return _loads(self._chat(system, user, schema, model=model, max_tokens=400))

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


def _maybe_thinking(model: str) -> bool:
    return any(t in model.lower() for t in ("qwen3", "deepseek-r1", "gpt-oss", "magistral", "phi4-reasoning"))


def make_backend(settings: ModelSettings, embed_model: str | None = None) -> Backend:
    check_offline(settings)  # files never leave your devices unless you opt out
    kinds = {"ollama": Ollama, "openai": OpenAICompatible}
    if settings.backend not in kinds:
        raise BackendError(f"unknown backend {settings.backend!r}; choose one of {', '.join(kinds)}")
    return kinds[settings.backend](settings, embed_model)
