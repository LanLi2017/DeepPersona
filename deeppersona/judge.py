"""LLM-judge for personas and research answers.

Each judge returns a scalar in [0, 1] plus a free-text critique. The critique is
the load-bearing signal for GEPA's reflective mutation — vague critiques produce
vague mutations, so the rubrics push the judge toward specific observations.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .bedrock_client import GenConfig, complete

_JUDGE_SYSTEM = (
    "You are a careful evaluator. Reply with a single JSON object and nothing else:\n"
    '{"score": <float in [0,1]>, "critique": "<2-5 sentences naming specific strengths and weaknesses, '
    'with quoted phrases from the artifact when possible>"}'
)

_PERSONA_RUBRIC = """Evaluate the persona on:
- coherence: traits, background, and stated constraints fit together without contradiction
- specificity: concrete details (career stage, expertise, motivations) rather than generic descriptors
- realism: plausible real person, not a caricature or a list of buzzwords
- task-fit: the persona's perspective is genuinely useful for the research query

Seed:
{seed}

Persona:
{persona}

Downstream research query the persona will answer:
{query}
"""

_RESEARCH_RUBRIC = """Evaluate the research answer on:
- groundedness: claims are specific and verifiable, not hand-wavy
- depth: addresses the query substantively, not at surface level
- persona-consistency: tone, framing, and emphasis match the stated persona's role and constraints
- usefulness: would actually help someone in the persona's situation make a decision

Persona:
{persona}

Research query:
{query}

Answer:
{answer}
"""


@dataclass(frozen=True)
class Judgement:
    score: float
    critique: str


def _parse(raw: str) -> Judgement:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return Judgement(0.0, f"Judge returned no JSON. Raw: {raw[:200]}")
    try:
        obj = json.loads(match.group(0))
    except ValueError as e:
        return Judgement(0.0, f"Judge JSON parse failed ({e}). Raw: {raw[:200]}")
    score = float(obj.get("score", 0.0))
    return Judgement(max(0.0, min(1.0, score)), str(obj.get("critique", "")))


def judge_persona(seed: str, persona: str, query: str, cfg: GenConfig) -> Judgement:
    return _parse(
        complete(_JUDGE_SYSTEM, _PERSONA_RUBRIC.format(seed=seed, persona=persona, query=query), cfg)
    )


def judge_research(persona: str, query: str, answer: str, cfg: GenConfig) -> Judgement:
    return _parse(
        complete(_JUDGE_SYSTEM, _RESEARCH_RUBRIC.format(persona=persona, query=query, answer=answer), cfg)
    )
