"""GEPA adapter that jointly optimizes the persona prompt and the downstream
research prompt against persona quality + research quality.

Components GEPA can mutate:
  - "persona_prompt": system prompt for generating a persona from a seed
  - "research_prompt": system prompt for answering a research query in-character

Each rollout: seed -> persona (student LM) -> answer (student LM) -> judge x2.
Trajectory captures every intermediate string + both judge critiques so the
reflection LM has concrete failures to reason over.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from .bedrock_client import GenConfig, complete
from .judge import Judgement, judge_persona, judge_research
from .seeds import Seed

_STUDENT_MODEL = os.environ.get(
    "DEEPPERSONA_STUDENT_MODEL", "mistral.ministral-3-3b-instruct"
)
_JUDGE_MODEL = os.environ.get(
    "DEEPPERSONA_JUDGE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

COMPONENTS = ("persona_prompt", "research_prompt")


@dataclass
class PersonaTrajectory:
    seed: dict
    persona: str
    research_query: str
    research_answer: str
    persona_judge: Judgement
    research_judge: Judgement


def _seed_brief(seed: Seed) -> str:
    return f"Topic: {seed.topic}\nRole: {seed.role}\nConstraints: {seed.constraints}"


class DeepPersonaAdapter(GEPAAdapter[Seed, PersonaTrajectory, dict]):
    def __init__(
        self,
        student_cfg: GenConfig | None = None,
        judge_cfg: GenConfig | None = None,
        persona_weight: float = 0.4,
        research_weight: float = 0.6,
    ):
        self.student_cfg = student_cfg or GenConfig(
            model_id=_STUDENT_MODEL, max_tokens=768, temperature=0.7
        )
        self.judge_cfg = judge_cfg or GenConfig(
            model_id=_JUDGE_MODEL, max_tokens=512, temperature=0.0
        )
        self.persona_weight = persona_weight
        self.research_weight = research_weight

    def _rollout(self, seed: Seed, candidate: dict[str, str]) -> PersonaTrajectory:
        brief = _seed_brief(seed)

        persona = complete(
            system=candidate["persona_prompt"], user=brief, cfg=self.student_cfg
        ).strip()

        research_user = f"Persona:\n{persona}\n\nResearch query:\n{seed.research_query}"
        answer = complete(
            system=candidate["research_prompt"], user=research_user, cfg=self.student_cfg
        ).strip()

        p_judge = judge_persona(brief, persona, seed.research_query, self.judge_cfg)
        r_judge = judge_research(persona, seed.research_query, answer, self.judge_cfg)

        return PersonaTrajectory(
            seed=asdict(seed),
            persona=persona,
            research_query=seed.research_query,
            research_answer=answer,
            persona_judge=p_judge,
            research_judge=r_judge,
        )

    def evaluate(
        self,
        batch: list[Seed],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PersonaTrajectory, dict]:
        trajectories: list[PersonaTrajectory] = []
        outputs: list[dict] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []

        for seed in batch:
            traj = self._rollout(seed, candidate)
            trajectories.append(traj)
            outputs.append(
                {"seed": traj.seed, "persona": traj.persona, "answer": traj.research_answer}
            )
            p, r = traj.persona_judge.score, traj.research_judge.score
            scores.append(self.persona_weight * p + self.research_weight * r)
            objective_scores.append({"persona": p, "research": r})

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[PersonaTrajectory, dict],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = eval_batch.trajectories or []
        out: dict[str, list[dict[str, Any]]] = {c: [] for c in components_to_update}

        for traj in trajectories:
            if "persona_prompt" in out:
                out["persona_prompt"].append(
                    {
                        "Inputs": {
                            "seed": traj.seed,
                            "downstream_query": traj.research_query,
                        },
                        "Generated Outputs": {"persona": traj.persona},
                        "Feedback": (
                            f"Persona judge score: {traj.persona_judge.score:.2f}/1.00\n"
                            f"Persona critique: {traj.persona_judge.critique}\n\n"
                            f"Downstream research score with this persona: "
                            f"{traj.research_judge.score:.2f}/1.00 — a vague or off-target persona "
                            f"often shows up as a weak downstream answer.\n"
                            f"Downstream critique: {traj.research_judge.critique}"
                        ),
                    }
                )

            if "research_prompt" in out:
                out["research_prompt"].append(
                    {
                        "Inputs": {
                            "persona": traj.persona,
                            "research_query": traj.research_query,
                        },
                        "Generated Outputs": {"answer": traj.research_answer},
                        "Feedback": (
                            f"Research judge score: {traj.research_judge.score:.2f}/1.00\n"
                            f"Research critique: {traj.research_judge.critique}"
                        ),
                    }
                )

        return out
