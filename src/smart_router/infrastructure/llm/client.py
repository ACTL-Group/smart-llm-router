from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
from openai import OpenAI
from smart_router.config.settings import Settings
from smart_router.infrastructure.vram.manager import IVRAMManager


class ILLMClient(ABC):
    """Interface for LLM and Embedding inference."""

    @abstractmethod
    def get_embedding(self, text: str, model: str | None = None) -> np.ndarray:
        """Generate normalized L2 embedding vector for given text."""
        pass

    @abstractmethod
    def evaluate_slm_logprobs(
        self,
        query: str,
        system_prompt: str,
        model: str | None = None,
        max_tokens: int = 3,
    ) -> tuple[str, list[float]]:
        """Run classification query on SLM returning (content, logprobs_list)."""
        pass

    @abstractmethod
    def generate_chat_completion(
        self,
        messages: Sequence[dict[str, str]],
        model: str,
        temperature: float = 0.2,
    ) -> str:
        """Run standard chat completion."""
        pass


class OpenAILLMClient(ILLMClient):
    """OpenAI API-compatible LLM Client for LocalAI or OpenAI endpoints."""

    def __init__(
        self,
        settings: Settings,
        vram_manager: IVRAMManager | None = None,
        client: OpenAI | None = None,
    ):
        self.settings = settings
        self.vram_manager = vram_manager
        self._client = client or OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.api_key,
        )

    @property
    def raw_client(self) -> OpenAI:
        return self._client

    def get_embedding(self, text: str, model: str | None = None) -> np.ndarray:
        model_name = model or self.settings.model_embed
        resp = self._client.embeddings.create(model=model_name, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def evaluate_slm_logprobs(
        self,
        query: str,
        system_prompt: str,
        model: str | None = None,
        max_tokens: int = 3,
    ) -> tuple[str, list[float]]:
        model_name = model or self.settings.model_cheap

        def _call_api():
            return self._client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                logprobs=True,
            )

        if self.vram_manager:
            with self.vram_manager.session(model_name):
                resp = _call_api()
        else:
            resp = _call_api()

        choice = resp.choices[0]
        token_logprobs = [
            item.logprob
            for item in (choice.logprobs.content or [])
            if item.logprob is not None
        ]
        raw_text = (choice.message.content or "").strip()
        return raw_text, token_logprobs

    def generate_chat_completion(
        self,
        messages: Sequence[dict[str, str]],
        model: str,
        temperature: float = 0.2,
    ) -> str:
        def _call_api():
            return self._client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
            )

        if self.vram_manager:
            with self.vram_manager.session(model):
                resp = _call_api()
        else:
            resp = _call_api()

        return resp.choices[0].message.content or ""
