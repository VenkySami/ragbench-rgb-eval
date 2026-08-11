"""Unified interface over 3 free-tier LLM APIs, with on-disk response caching so
reruns/aggregation are free and deterministic.

None of the 3 models is OpenAI GPT / ChatGPT / ChatGLM (the models referenced in
the task PDF and the RGB reference repo) -- all 3 have no-credit-card free tiers.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_random_exponential

from . import config

try:
    from dotenv import load_dotenv
    load_dotenv(config.ROOT / ".env")
except ImportError:
    pass


class LLMError(RuntimeError):
    pass


class BaseLLMClient(ABC):
    name: str
    min_interval_seconds: float = 0.0  # per-client throttle for strict free-tier RPM limits

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        wait = self.min_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    @abstractmethod
    def _call(self, system: str, user: str, temperature: float) -> str:
        ...

    def generate(self, system: str, user: str, temperature: float = 0.2, use_cache: bool = True) -> str:
        cache_key = self._cache_key(system, user, temperature)
        cache_path = config.CACHE_DIR / f"{cache_key}.json"
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text())["response"]

        self._throttle()
        response = self._call(system, user, temperature)

        if use_cache:
            cache_path.write_text(json.dumps({
                "model": self.name,
                "model_id": self.model_id,
                "system": system,
                "user": user,
                "response": response,
            }))
        return response

    def _cache_key(self, system: str, user: str, temperature: float) -> str:
        raw = f"{self.name}|{self.model_id}|{temperature}|{system}|{user}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class GeminiClient(BaseLLMClient):
    name = "gemini"
    min_interval_seconds = 13.0  # observed free tier: gemini-flash-latest ~5 req/min

    def __init__(self, model_id: str = config.MODELS["gemini"]):
        super().__init__(model_id)
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY not set (see .env.example)")
        self._client = genai.Client(api_key=api_key)

    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(8))
    def _call(self, system: str, user: str, temperature: float) -> str:
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
            ),
        )
        return (resp.text or "").strip()


class GroqClient(BaseLLMClient):
    name = "groq"
    min_interval_seconds = 2.5  # groq free tier is generous but still rate-limited

    def __init__(self, model_id: str = config.MODELS["groq"]):
        super().__init__(model_id)
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMError("GROQ_API_KEY not set (see .env.example)")
        self._client = Groq(api_key=api_key)

    @retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(5))
    def _call(self, system: str, user: str, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class MistralClient(BaseLLMClient):
    name = "mistral"
    min_interval_seconds = 2.0  # Mistral free tier: 1 req/sec

    def __init__(self, model_id: str = config.MODELS["mistral"]):
        super().__init__(model_id)
        try:
            from mistralai import Mistral  # older SDK layout
        except ImportError:
            from mistralai.client import Mistral  # SDK >=2.x layout
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise LLMError("MISTRAL_API_KEY not set (see .env.example)")
        self._client = Mistral(api_key=api_key)

    @retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(5))
    def _call(self, system: str, user: str, temperature: float) -> str:
        resp = self._client.chat.complete(
            model=self.model_id,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


CLIENT_REGISTRY = {
    "gemini": GeminiClient,
    "groq": GroqClient,
    "mistral": MistralClient,
}


def get_client(name: str) -> BaseLLMClient:
    if name not in CLIENT_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(CLIENT_REGISTRY)}")
    return CLIENT_REGISTRY[name]()
