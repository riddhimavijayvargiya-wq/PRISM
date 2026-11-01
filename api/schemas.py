from pydantic import BaseModel
from typing import Optional


class RagPlaygroundRequest(BaseModel):
    question: str
    model: str
    pipelines: Optional[list[str]] = None  # None = use all 3 (A, B, C)


class LlmPlaygroundRequest(BaseModel):
    question: str
    pipeline: str
    models: list[str]