"""Axis-structured evolutionary system-prompt optimization (E-SPL port, evolution-only).

Adapts E-SPL's genetic prompt evolution (refs/E-SPL/.../system_prompt_learning_rl.py,
the train_mode="evolution" path — no gradient/RL) in two ways for this experiment:

  1. Programs hold principles keyed by a *predefined persona-axis* template
     (role / method / verification / pitfalls / other) instead of E-SPL's flat
     G0..Gn list. render_system_prompt groups them under axis headings.
  2. Mutation & crossover are axis-aware: every add/modify op carries an `axis`, so
     the evolved system prompt stays organized by axis; "other" is the catch-all for
     general takeaways that fit no axis.

LLM self-reflection (mutation + crossover) runs through OpenAI (OpenAIChat).
Rollouts / fitness are supplied by the caller (Tinker). TrueSkill (trueskill_utils)
drives selection. We streamline E-SPL's 3-stage mutation to: per-problem axis-aware
critique -> apply ops -> one merge/dedup pass (we skip E-SPL's per-rollout summary
LLM call and instead truncate trajectories — flagged; can be reinstated if needed).
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from deeppersona.espl_json import fix_json_backslashes, remove_json_comments
from deeppersona.trueskill_utils import Rating

# ── axis template (the "predefined persona axis-based template") ───────────
_AXIS_SETS = {
    "persona5": (
        ["role", "method", "verification", "pitfalls", "other"],
        {
            "role": "Role / mindset the solver adopts (who they are, the disposition they bring).",
            "method": "Solution method: how to structure the approach (decompose, set up equations, work backwards, casework, exploit symmetry, ...).",
            "verification": "Verification habits: concrete self-checks before committing (substitute back, check bounds/parity/units, recompute a critical step, sanity-check the magnitude).",
            "pitfalls": "Common pitfalls to avoid: recurring failure modes seen in the attempts (sign/arithmetic slips, mis-reading the question, unjustified guesses, dropped cases).",
            "other": "Other: a genuinely general, cross-cutting takeaway that fits none of the axes above. Use sparingly.",
        },
    ),
    "compmath": (  # task-specific axes for competition math (AIME-style)
        ["setup", "technique", "casework", "computation", "verification", "other"],
        {
            "setup": "Reading & re-encoding the problem: pick good variables/coordinates, restate constraints precisely, name the key quantity or invariant, and exploit symmetry before computing.",
            "technique": "Which competition method to reach for, by domain: modular arithmetic / CRT (number theory); complementary & bijective counting (combinatorics); Vieta's, factoring, substitution (algebra); coordinates, trig, power-of-a-point (geometry); generating functions, recursion, telescoping.",
            "casework": "Enumeration discipline: partition into exhaustive, disjoint cases; bound the search space first; track which cases remain; avoid double-counting and dropped cases.",
            "computation": "Exact-arithmetic control: keep fractions/radicals exact, simplify before expanding, defer numeric evaluation, and re-check heavy multiplications or large factorials.",
            "verification": "Answer-form checks: the final answer is a non-negative integer (AIME: 0-999) — verify it lands in the expected range; substitute the solution back into every constraint; confirm against a small case, parity, or a modular residue.",
            "other": "A genuinely cross-cutting takeaway that fits none of the axes above. Use sparingly.",
        },
    ),
    "flat": (  # ablation: minimal structure, closest to flat E-SPL
        ["strategy", "other"],
        {
            "strategy": "A useful strategy, heuristic, or principle for solving the problem.",
            "other": "A general takeaway that is not a concrete solving strategy.",
        },
    ),
}

AXES, AXIS_GUIDE = _AXIS_SETS["persona5"]
ANSWER_INSTRUCTION = "Put your final answer inside \\boxed{}."


def configure_axes(name: str) -> None:
    """Rebind the active axis template (call once at startup)."""
    global AXES, AXIS_GUIDE
    AXES, AXIS_GUIDE = _AXIS_SETS[name]


def _norm_axis(a) -> str:
    return a if a in AXES else AXES[-1]  # fall back to the catch-all ("other")


def _axis_guide_str() -> str:
    return "\n".join(f"- {a}: {AXIS_GUIDE[a]}" for a in AXES)


# ── principle dict: {Gi: {"axis": ax, "text": txt}} ────────────────────────
def _next_id(principles: dict) -> int:
    return max((int(k[1:]) for k in principles if k[1:].isdigit()), default=-1) + 1


def _rekey(principles: dict) -> dict:
    """Reassign G0..Gn in axis order for stable, readable ids."""
    by_axis = defaultdict(list)
    for p in principles.values():
        by_axis[_norm_axis(p["axis"])].append(p["text"])
    out, i = {}, 0
    for ax in AXES:
        for txt in by_axis[ax]:
            out[f"G{i}"] = {"axis": ax, "text": txt}
            i += 1
    return out


def render_system_prompt(principles: dict) -> str:
    """System message the policy is conditioned on."""
    if not principles:
        return "Solve the problem. " + ANSWER_INSTRUCTION
    by_axis = defaultdict(list)
    for pid, p in principles.items():
        by_axis[_norm_axis(p["axis"])].append((pid, p["text"]))
    lines = ["When solving the problem, carefully follow the principles below, organized by aspect:\n"]
    for ax in AXES:
        if by_axis[ax]:
            lines.append(f"## {ax.capitalize()}")
            lines += [f"- [{pid}] {txt}" for pid, txt in by_axis[ax]]
            lines.append("")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


def format_principles(principles: dict, label: str = "") -> str:
    if not principles:
        return "None"
    pre = f"{label}." if label else ""
    by_axis = defaultdict(list)
    for pid, p in principles.items():
        by_axis[_norm_axis(p["axis"])].append((pid, p["text"]))
    out = []
    for ax in AXES:
        for pid, txt in by_axis[ax]:
            out.append(f"[{pre}{pid}] (axis={ax}): {txt}")
    return "\n".join(out)


# ── OpenAI self-reflection client ──────────────────────────────────────────
class OpenAIChat:
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 4000, max_workers: int = 8):
        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # reasoning models (o*/gpt-5*) reject temperature/max_tokens and burn the
        # completion budget on hidden reasoning -> use max_completion_tokens with headroom.
        self.is_reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.n_calls = 0

    def __call__(self, prompt: str) -> str:
        self.n_calls += 1
        msgs = [{"role": "user", "content": prompt}]
        if self.is_reasoning:
            r = self.client.chat.completions.create(
                model=self.model, messages=msgs,
                max_completion_tokens=max(self.max_tokens, 16000))
        else:
            r = self.client.chat.completions.create(
                model=self.model, messages=msgs,
                temperature=self.temperature, max_tokens=self.max_tokens)
        return r.choices[0].message.content or ""

    def map(self, prompts: list[str]) -> list[str]:
        return list(self.pool.map(self.__call__, prompts))


# ── templates (sentinel placeholders -> fill(); avoids str.format brace hell) ─
def fill(tmpl: str, **kw) -> str:
    for k, v in kw.items():
        tmpl = tmpl.replace(f"<<{k}>>", str(v))
    return tmpl


CRITIQUE_TEMPLATE = """An agent solved a math problem several times under a set of axis-organized principles, producing a mix of correct and wrong attempts. Extract generalizable, axis-organized lessons that would make future attempts more reliable.

The principles are organized under these fixed axes:
<<axis_guide>>

Steps:
1. Compare the correct vs wrong attempts: what decision or habit made the difference? Did a missing or weak principle let the wrong attempts fail?
2. Propose at most <<max_operations>> edits. Operations: "add" (introduce a new principle) or "modify" (improve an existing one; cite its id in modified_from).
3. Each principle MUST: be a generalizable strategic lesson (not arithmetic specific to this one problem), be <= 32 words, and be tagged with exactly one axis from <<axis_list>>. Use "other" only for genuinely cross-cutting takeaways.

Reason step by step first, then output ONLY a JSON list at the very end:
```json
[
  {"operation": "add", "axis": "verification", "principle": "..."},
  {"operation": "modify", "axis": "pitfalls", "modified_from": "G3", "principle": "..."}
]
```

<problem>
<<problem>>
</problem>

<attempts>
<<attempts>>
</attempts>

<groundtruth>
<<answer>>
</groundtruth>

<current_principles>
<<principles>>
</current_principles>"""


MERGE_TEMPLATE = """Below is a set of axis-organized principles for solving math problems. Some may be near-duplicates or overlapping. Produce a clean revision plan that removes redundancy while preserving every distinct, useful idea.

Requirements for each resulting principle: clear and generalizable, <= 32 words, tagged with exactly one axis from <<axis_list>>.

Operations: "modify" (tighten/clarify one principle; cite modified_from) and "merge" (combine >=2 near-duplicate principles into one; cite merged_from). Keep edits minimal — if nothing needs changing, return an empty list.

Reason briefly, then output ONLY a JSON list at the end:
```json
[
  {"operation": "modify", "axis": "method", "modified_from": "G2", "principle": "..."},
  {"operation": "merge", "axis": "verification", "merged_from": ["G4", "G7"], "principle": "..."}
]
```

<principles>
<<principles>>
</principles>"""


CROSSOVER_START = """You are given multiple sets of axis-organized principles that were independently evolved to guide an agent in solving math problems. Principle Set A is the primary set we want to improve by learning from the strengths of the others.
"""

CROSSOVER_ANALYSIS = """Each principle set above strictly outperformed the others on at least one question. Analyze this evidence and improve Principle Set A.

Steps:
1. For each question, explain WHY the best-performing set likely succeeded — cite principle ids (e.g. A.G3, B.G1) and question ids (e.g. Q.0). Distinguish targeted guidance from luck.
2. Decide what to preserve in Set A and what to import from other sets. Any addition/modification to Set A must be inspired by a principle that helped another set, not invented from scratch.
3. Keep axis tags consistent; each resulting principle is <= 32 words and tagged with exactly one axis.

Operations on Set A: "add", "modify" (cite modified_from), "remove" (cite remove_id). Only propose changes clearly supported by the evidence; an empty list is acceptable.

Reason step by step, then output ONLY a JSON list at the end:
```json
[
  {"operation": "add", "axis": "method", "evidence": "Set B won Q.3 via B.G1", "principle": "..."},
  {"operation": "modify", "axis": "pitfalls", "modified_from": "A.G2", "evidence": "...", "principle": "..."},
  {"operation": "remove", "remove_id": "A.G5", "evidence": "..."}
]
```"""


def build_crossover_prompt(sets: list[dict], problems_each_best: list[list]) -> str:
    parts = [CROSSOVER_START]
    for i, pr in enumerate(sets):
        label = chr(ord("A") + i)
        parts += [f"\nPrinciple Set {label}:", "<principles>", format_principles(pr, label), "</principles>"]
    parts.append("\nBelow are the questions each principle set performed best on.")
    for i, probs in enumerate(problems_each_best):
        label = chr(ord("A") + i)
        parts += [f"\nQuestions Set {label} did best on:", "<questions>"]
        parts += [f"Q.{qid}: {q.strip()[:400]}" for qid, q in probs]
        parts.append("</questions>")
    parts.append("\n" + fill(CROSSOVER_ANALYSIS, axis_list=AXES))
    return "\n".join(parts)


# ── op parsing + application ───────────────────────────────────────────────
def parse_ops(text: str) -> list:
    chunk = text.split("```json")[-1].split("```")[0]
    for tf in (lambda s: s, fix_json_backslashes, lambda s: remove_json_comments(fix_json_backslashes(s))):
        try:
            ops = json.loads(tf(chunk))
            return ops if isinstance(ops, list) else []
        except Exception:
            continue
    return []


def _apply_ops(principles: dict, ops: list):
    nid = _next_id(principles)
    for op in ops:
        try:
            kind = op.get("operation")
            if kind == "add":
                principles[f"G{nid}"] = {"axis": _norm_axis(op.get("axis")), "text": op["principle"].strip()}
                nid += 1
            elif kind == "modify":
                pid = str(op["modified_from"]).split(".")[-1]
                if pid in principles:
                    principles[pid] = {"axis": _norm_axis(op.get("axis", principles[pid]["axis"])),
                                       "text": op["principle"].strip()}
            elif kind == "merge":
                src = [s for s in (str(x).split(".")[-1] for x in op.get("merged_from", [])) if s in principles]
                if len(src) >= 2:
                    ax = _norm_axis(op.get("axis", principles[src[0]]["axis"]))
                    for s in src:
                        del principles[s]
                    principles[f"G{nid}"] = {"axis": ax, "text": op["principle"].strip()}
                    nid += 1
            elif kind == "remove":
                principles.pop(str(op.get("remove_id", "")).split(".")[-1], None)
        except Exception as e:
            print(f"[evo] skipped bad op {op}: {e}", flush=True)
    return principles


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    head = n // 3
    return s[:head] + "\n...[truncated]...\n" + s[-(n - head):]


# ── genetic operators ──────────────────────────────────────────────────────
def mutate(parent_principles: dict, rollouts: list, llm: OpenAIChat,
           max_ops: int = 2, max_problems: int = 6, max_attempts: int = 5, trunc: int = 2000):
    """rollouts: [{problem, answer, attempts:[{text, correct}]}] for the best program."""
    def acc(r):
        return sum(a["correct"] for a in r["attempts"]) / max(len(r["attempts"]), 1)

    mixed = [r for r in rollouts if 0 < acc(r) < 1]
    if not mixed:  # no within-problem signal -> reflect on all-wrong problems instead
        mixed = [r for r in rollouts if acc(r) == 0] or rollouts
    random.shuffle(mixed)
    mixed = mixed[:max_problems]

    prompts = []
    for r in mixed:
        atts = r["attempts"][:max_attempts]
        att_str = "\n\n".join(
            f"Attempt {i + 1} ({'correct' if a['correct'] else 'wrong'}):\n{_truncate(a['text'], trunc)}"
            for i, a in enumerate(atts))
        prompts.append(fill(CRITIQUE_TEMPLATE, axis_guide=_axis_guide_str(), axis_list=AXES,
                            max_operations=max_ops, problem=r["problem"], attempts=att_str,
                            answer=r["answer"], principles=format_principles(parent_principles)))
    ops = []
    for resp in llm.map(prompts):
        ops.extend(parse_ops(resp)[:max_ops])

    principles = _apply_ops(copy.deepcopy(parent_principles), ops)
    if principles:  # merge/dedup pass
        merge_ops = parse_ops(llm(fill(MERGE_TEMPLATE, axis_list=AXES,
                                       principles=format_principles(principles))))
        principles = _apply_ops(principles, merge_ops)
    return _rekey(principles), {"critique_ops": ops, "n_problems": len(mixed)}


def crossover(parents_principles: list[dict], problems_each_best: list[list], llm: OpenAIChat):
    """parents_principles[0] = Set A (the top program), to be improved."""
    resp = llm(build_crossover_prompt(parents_principles, problems_each_best))
    principles = _apply_ops(copy.deepcopy(parents_principles[0]), parse_ops(resp))
    return _rekey(principles), {"response_chars": len(resp)}


# ── rating helpers (from E-SPL calculate_*_rating) ─────────────────────────
def mutation_rating(r: Rating, sigma: float = 1.0) -> Rating:
    return Rating(r.mu, math.sqrt(r.sigma ** 2 + sigma ** 2))


def crossover_rating(ratings: list[Rating], sigma: float = 1.0) -> Rating:
    prec = sum(x.precision for x in ratings)
    mu = sum(x.precision_mean for x in ratings) / prec
    return Rating(mu, math.sqrt(1.0 / prec + sigma ** 2))


# ── population ─────────────────────────────────────────────────────────────
class AxisProgram:
    def __init__(self, principles, program_id, origin="root", parent=-1, parents_list=None,
                 mu=25.0, sigma=25.0 / 3, timestep=None):
        self.principles = principles
        self.program_id = program_id
        self.origin = origin  # root | mutation | crossover
        self.parent = parent
        self.parents_list = parents_list or []
        self.children = []
        self.history = []  # per-step fitness (pass@1) when this program was rolled out
        self.rating = Rating(mu, sigma)
        self.timestep = timestep

    def explore_score(self, lam=2.0):
        return self.rating.mu + lam * self.rating.sigma

    def state_dict(self):
        return {"program_id": self.program_id, "origin": self.origin, "parent": self.parent,
                "parents_list": self.parents_list, "children": self.children,
                "timestep": self.timestep, "history": self.history,
                "rating": self.rating.state_dict(), "principles": self.principles}


class EvolutionPool:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.programs = []
        self._next = 0

    def new_id(self):
        i = self._next
        self._next += 1
        return i

    def add(self, prog: AxisProgram):
        self.programs.append(prog)
        if len(self.programs) > self.max_size:
            self.programs = self.programs[-self.max_size:]

    def select(self, m, recent_k=5, strategy="ucb", lam=2.0, rng=None):
        rng = rng or random
        pool = self.programs[-recent_k:] if recent_k else list(self.programs)
        if len(pool) <= m:
            return list(pool)
        if strategy == "ucb":
            return sorted(pool, key=lambda p: p.explore_score(lam), reverse=True)[:m]
        chosen = [pool[-1]]  # uniform: always include the most recent
        chosen += rng.sample([p for p in pool if p is not pool[-1]], m - 1)
        return chosen

    def best(self, by="mean"):
        if by == "mean":
            return max(self.programs, key=lambda p: p.rating.mu)
        return max(self.programs, key=lambda p: (sum(p.history) / len(p.history)) if p.history else -1)
