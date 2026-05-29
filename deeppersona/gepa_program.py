"""DSPy program + GEPA feedback metric for persona-prompt optimization.

The research question (see docs/q1_progress.md §4-5): the *structured* persona
prompts under-performed *basic* on CSQA. Is that because structure is
intrinsically costly at 7B, or because the hand-written structured text is
*under-optimized*? GEPA (Genetic-Pareto reflective prompt evolution) is the
tool to test it: we seed a predictor's instruction with a structured persona
and let GEPA rewrite it, then measure whether the optimized prompt recovers.

What GEPA optimizes here is the *instruction* (system prompt) of a single
`dspy.Predict`. The answer-format requirement (a boxed A-E letter) lives in the
OutputField description so it stays fixed across GEPA's rewrites — this keeps
the search aligned with the native-harness verdict, which appends the same
boxed-letter instruction.

NOTE: scoring during the GEPA search reuses the *exact* CSQA verifier from
`verifiers.py`, so the metric is identical to the headline pipeline. Only the
prompt *formatting* (DSPy's chat adapter vs. our `apply_chat_template`) differs,
which is why the final verdict is re-measured in the native harness
(`scripts/06_gepa_eval.py`).
"""
from __future__ import annotations

import dspy

from .data import load_items
from .personas import ANSWER_INSTRUCTION, structured_persona, system_message
from .verifiers import extract_pred, score


# ── DSPy signature ─────────────────────────────────────────────────────────
# The docstring is a placeholder; the real instruction is injected at build
# time (seeded from a persona) and then evolved by GEPA. The OutputField desc
# carries the format contract and is held fixed.
class CSQASignature(dspy.Signature):
    """Answer the commonsense multiple-choice question."""

    question: str = dspy.InputField(
        desc="A commonsense question followed by options labeled A through E."
    )
    answer: str = dspy.OutputField(
        desc="Reason briefly, then end with the single letter (A, B, C, D, or E) "
        "of the best option inside \\boxed{}."
    )


def build_program(task: str, persona_idx: int, level: str = "structured") -> dspy.Predict:
    """A `dspy.Predict` whose instruction is seeded from a persona prompt.

    level="structured" -> the multi-section block (the under-performing prompt
    we want to optimize). level="basic" -> the one-line framing. The seed is the
    persona body only; the boxed-letter format lives in the OutputField.
    """
    if level == "structured":
        seed = structured_persona(task, persona_idx)
    elif level == "basic":
        seed = TASK_SPECS_BASIC(task, persona_idx)
    else:
        raise ValueError(f"unknown seed level: {level}")
    sig = CSQASignature.with_instructions(seed)
    return dspy.Predict(sig)


def TASK_SPECS_BASIC(task: str, persona_idx: int) -> str:
    # system_message(level="basic") appends ANSWER_INSTRUCTION; strip it back off
    # so the seed is the persona body only (format lives in the OutputField).
    msg = system_message(task, persona_idx, 0, "basic")
    return msg[: -len(ANSWER_INSTRUCTION[task])] if msg.endswith(ANSWER_INSTRUCTION[task]) else msg


def build_examples(task: str, split: str, n_items: int | None = None, seed: int = 0) -> list[dspy.Example]:
    """Frozen-split items as DSPy examples (input=question, gold=letter/number)."""
    items = load_items(task, split, n_items=n_items, seed=seed)
    out = []
    for it in items:
        ex = dspy.Example(question=it["question"], answer=it["gold_answer"], idx=it["idx"])
        out.append(ex.with_inputs("question"))
    return out


# ── GEPA feedback metric ────────────────────────────────────────────────────
def make_metric(task: str):
    """Returns a GEPA metric(gold, pred, trace, pred_name, pred_trace).

    Score = exact-match (1/0) via the native verifier. Feedback is textual and
    diagnostic: it reports the parsed letter, the gold answer, and the response
    length (we found longer CSQA outputs correlate with errors), so the
    reflection LM has a concrete signal to act on. The feedback states facts,
    not a prescribed fix, so it does not pre-bake the conclusion.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        ans_text = (getattr(pred, "answer", "") or "")
        gold_ans = gold.answer
        pred_letter = extract_pred(task, ans_text)
        ok = bool(score(task, pred_letter, gold_ans))
        s = 1.0 if ok else 0.0
        nwords = len(ans_text.split())

        if task == "csqa":
            if ok:
                fb = f"Correct: selected {pred_letter} (gold {gold_ans}). Response was {nwords} words."
            elif pred_letter is None:
                fb = (
                    f"Incorrect: could not parse a single A-E letter. The gold answer is {gold_ans}. "
                    f"The response MUST end with exactly one option letter inside \\boxed{{}}. "
                    f"Response was {nwords} words."
                )
            else:
                fb = (
                    f"Incorrect: selected {pred_letter} but the gold answer is {gold_ans}. "
                    f"Reconsider which option best fits everyday common sense and commit to one. "
                    f"Response was {nwords} words."
                )
        else:  # gsm8k or other verifiable-number tasks
            if ok:
                fb = f"Correct: answer {pred_letter} matches gold {gold_ans}. Response was {nwords} words."
            else:
                fb = (
                    f"Incorrect: produced {pred_letter!r} but gold is {gold_ans}. "
                    f"Recompute carefully and put the final number in \\boxed{{}}. "
                    f"Response was {nwords} words."
                )
        return dspy.Prediction(score=s, feedback=fb)

    return metric


def get_instruction(program) -> str:
    """Extract the (possibly evolved) instruction from a Predict or Module."""
    if hasattr(program, "signature"):
        return program.signature.instructions
    # Module: return the first named predictor's instruction.
    for _, p in program.named_predictors():
        return p.signature.instructions
    raise ValueError("could not locate a predictor instruction on program")
