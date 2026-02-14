from __future__ import annotations

import os
import json
from typing import Protocol, Type, TypeVar, cast
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


class OpenAILLM:
    """
    OpenAI official SDK wrapper.

    - complete(): returns a JSON string (JSON mode). Good for simple cases but does NOT guarantee schema.
    - complete_structured(): returns a parsed Pydantic model using Structured Outputs (recommended).
    """

    def __init__(self, model: str | None = None):
        self.client = OpenAI()  # reads OPENAI_API_KEY from env by default
        # Recommended: set OPENAI_MODEL=gpt-5-mini in your .env
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")

    def complete(self, system: str, user: str) -> str:
        """
        JSON mode: guarantees valid JSON, not strict schema adherence.
        Prefer complete_structured() when you need schema-correct outputs.
        """
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text={"format": {"type": "json_object"}},
        )
        out = resp.output_text
        if out is None:
            raise RuntimeError("OpenAI response output_text is None.")
        return out

    def complete_structured(self, system: str, user: str, model_cls: Type[T]) -> T:
        """
        Structured Outputs: the model is constrained to the schema derived from model_cls.
        Returns a Pydantic model instance.
        """
        resp = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=model_cls,
        )
        parsed = resp.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI response output_parsed is None.")
        return cast(T, parsed)
