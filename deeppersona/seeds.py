"""Built-in seed list used as the bootstrap eval set.

Each Seed is a `(topic, role, constraints, research_query)` tuple: the persona
prompt expands the first three into a persona, the research prompt then answers
the query in-character.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Seed:
    topic: str
    role: str
    constraints: str
    research_query: str


_SEEDS: list[Seed] = [
    Seed(
        topic="climate adaptation",
        role="municipal planner in a coastal city",
        constraints="mid-career, non-technical background, tight capital budget",
        research_query="What are the most cost-effective sea-level-rise mitigations for a city of 200k people over a 10-year horizon?",
    ),
    Seed(
        topic="rare-disease drug pricing",
        role="patient advocate",
        constraints="parent of an affected child, US-based, no medical training",
        research_query="How do US orphan-drug pricing models compare to the EU's, and what reforms are currently being proposed?",
    ),
    Seed(
        topic="open-weight AI governance",
        role="policy researcher at a think tank",
        constraints="PhD in political science, skeptical of industry self-regulation",
        research_query="Which enforcement mechanisms in the EU AI Act actually apply to open-weight model releases?",
    ),
    Seed(
        topic="industrial heat decarbonization",
        role="chief sustainability officer at a cement company",
        constraints="reports to a cost-focused CEO, capital cycles measured in decades",
        research_query="Which low-carbon clinker substitutes are commercially viable at scale within five years?",
    ),
    Seed(
        topic="postpartum mental health",
        role="rural family physician",
        constraints="solo practice, limited specialist referral access",
        research_query="What screening protocols and short-term interventions have the strongest evidence for postpartum depression in primary care?",
    ),
    Seed(
        topic="semiconductor export controls",
        role="supply-chain analyst at a mid-size fab equipment vendor",
        constraints="reports to legal, has to brief non-technical executives",
        research_query="How have the October 2023 US export-control updates reshaped lithography-tool sales pipelines into China?",
    ),
]


def default_seeds() -> list[Seed]:
    return list(_SEEDS)
