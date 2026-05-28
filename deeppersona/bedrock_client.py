"""Bedrock Converse API wrapper.

Converse gives one request/response shape across Mistral, Claude, Llama, etc.,
so the rest of the codebase never branches on model family.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import boto3
from botocore.config import Config

_REGION = os.environ.get("AWS_REGION", "us-east-2")


@dataclass(frozen=True)
class GenConfig:
    model_id: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


_client_cache: dict[str, "boto3.client"] = {}


def _client():
    if _REGION not in _client_cache:
        _client_cache[_REGION] = boto3.client(
            "bedrock-runtime",
            region_name=_REGION,
            config=Config(retries={"max_attempts": 5, "mode": "standard"}),
        )
    return _client_cache[_REGION]


def chat(
    system: str | None,
    messages: list[dict[str, str]],
    cfg: GenConfig,
) -> str:
    """messages: [{"role": "user"|"assistant", "content": "..."}, ...]"""
    kwargs: dict = {
        "modelId": cfg.model_id,
        "messages": [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
        ],
        "inferenceConfig": {
            "maxTokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "topP": cfg.top_p,
        },
    }
    if system:
        kwargs["system"] = [{"text": system}]
    resp = _client().converse(**kwargs)
    return resp["output"]["message"]["content"][0]["text"]


def complete(system: str | None, user: str, cfg: GenConfig) -> str:
    return chat(system, [{"role": "user", "content": user}], cfg)
