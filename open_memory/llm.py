"""A minimal OpenAI-compatible chat client (OpenAI, Groq, OpenRouter, Ollama, vLLM...).

No SDK dependency. Retries on 429/5xx and honours retry-after, because a memory
write that silently drops on a rate limit is how memories go missing.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

# Bring your own key: every provider below speaks the OpenAI chat format.
# The key is read from the named environment variable; nothing is ever sent anywhere else.
PRESETS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),  # fully local, no key
}


def from_spec(spec: str | None) -> "LLM | None":
    """'provider:model', e.g. 'anthropic:claude-sonnet-5-5' or 'ollama:llama3.2'. None/'' means no LLM."""
    if not spec:
        return None
    provider, _, model = spec.partition(":")
    if provider not in PRESETS:
        raise ValueError(f"unknown provider {provider!r}; pick one of {', '.join(PRESETS)}")
    if not model:
        raise ValueError(f"name a model after the colon, e.g. {provider}:<model>")
    env = PRESETS[provider][1]
    if env and not os.environ.get(env):
        raise ValueError(f"{provider} needs {env} set in your environment")
    return LLM(model, provider=provider)


class LLM:
    def __init__(self, model: str, provider: str = "openai", base_url: str | None = None,
                 api_key: str | None = None, max_retries: int = 8):
        if provider not in PRESETS and not base_url:
            raise ValueError(f"unknown provider {provider!r}; pass base_url for a custom endpoint")
        base, env = PRESETS.get(provider, (base_url, None))
        self.base = (base_url or base).rstrip("/")
        self.key = api_key or (os.environ.get(env) if env else None) or "none"
        self.model = model
        self.max_retries = max_retries
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def chat(self, messages: list[dict] | str, max_tokens: int = 1024, temperature: float = 0.0, **extra) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, **extra}
        for attempt in range(self.max_retries):
            req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.key}",
                                                  # Cloudflare-fronted APIs reject urllib's default UA (1010)
                                                  "User-Agent": "open-memory/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = json.load(r)
                u = data.get("usage") or {}
                self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                self.usage["calls"] += 1
                return (data["choices"][0]["message"].get("content") or "").strip()
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait = e.headers.get("retry-after")
                    try:
                        wait = float(wait)
                    except (TypeError, ValueError):
                        wait = min(60, 2 ** attempt * 2)
                    time.sleep(wait + 0.5)
                    continue
                raise RuntimeError(f"LLM {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError):
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("LLM retries exhausted")

    def json(self, prompt: str, max_tokens: int = 4096, **extra):
        try:
            text = self.chat(prompt, max_tokens=max_tokens, response_format={"type": "json_object"}, **extra)
        except RuntimeError as e:  # some providers reject json mode mid-generation; parse plain text instead
            if "json_validate_failed" not in str(e):
                raise
            text = self.chat(prompt + "\n\nReturn only the JSON object.", max_tokens=max_tokens, **extra)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise RuntimeError(f"LLM returned no JSON: {text[:200]!r}")
        return json.loads(m.group(0))
