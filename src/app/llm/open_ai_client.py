from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.getcwd(), "_secret", ".env"))


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    chat_model: str
    embed_model: str


def load_config() -> OpenAIConfig:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não encontrado. Crie _secret/.env e defina OPENAI_API_KEY=..."
        )

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

    return OpenAIConfig(api_key=api_key, chat_model=chat_model, embed_model=embed_model)


def build_client(cfg: Optional[OpenAIConfig] = None) -> OpenAI:
    cfg = cfg or load_config()
    return OpenAI(api_key=cfg.api_key)


def embed_texts(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Gera embeddings para uma lista de textos.
    Retorna uma lista de vetores (list[float]).
    """
    cfg = load_config()
    client = build_client(cfg)
    use_model = model or cfg.embed_model

    resp = client.embeddings.create(model=use_model, input=texts)
    return [item.embedding for item in resp.data]


def chat_complete(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_output_tokens: int = 180,
    temperature: float = 0.0,
) -> str:
    """
    Faz uma chamada de chat e retorna apenas o texto final.
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    cfg = load_config()
    client = build_client(cfg)
    use_model = model or cfg.chat_model

    resp = client.chat.completions.create(
        model=use_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    return (resp.choices[0].message.content or "").strip()