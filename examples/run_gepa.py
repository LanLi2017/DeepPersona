"""Optimize the persona + research prompts with GEPA on Bedrock."""
from __future__ import annotations

import os
from pathlib import Path

import gepa
from dotenv import load_dotenv

from deeppersona.bedrock_client import GenConfig, complete
from deeppersona.gepa_adapter import DeepPersonaAdapter
from deeppersona.seeds import default_seeds

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


SEED_PERSONA_PROMPT = (
    "You write detailed, plausible personas for downstream research tasks. "
    "Given a topic, role, and constraints, produce a 4-6 sentence persona including "
    "name, background, current situation, what they care about, and how they would "
    "approach the research query. Prose, not bullet points."
)

SEED_RESEARCH_PROMPT = (
    "You answer a research query in-character as the given persona. Stay in voice. "
    "Cite the persona's concrete constraints when shaping the answer. 250-400 words."
)


def _reflection_lm():
    cfg = GenConfig(
        model_id=os.environ.get(
            "DEEPPERSONA_REFLECTION_MODEL",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        max_tokens=2048,
        temperature=1.0,
    )

    def call(prompt: str) -> str:
        return complete(system=None, user=prompt, cfg=cfg)

    return call


def main():
    seeds = default_seeds()
    trainset = seeds[:3]
    valset = seeds[3:]

    result = gepa.optimize(
        seed_candidate={
            "persona_prompt": SEED_PERSONA_PROMPT,
            "research_prompt": SEED_RESEARCH_PROMPT,
        },
        trainset=trainset,
        valset=valset,
        adapter=DeepPersonaAdapter(),
        reflection_lm=_reflection_lm(),
        max_metric_calls=60,
        display_progress_bar=True,
    )

    print("\n=== Best candidate ===")
    for k, v in result.best_candidate.items():
        print(f"\n--- {k} ---\n{v}")


if __name__ == "__main__":
    main()
