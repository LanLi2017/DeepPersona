# LLM judge outputs

These files contain post-hoc pre-annotations from three independent judges.
They are not part of the blind annotation input and must remain unopened until
an annotator has exported independent labels from
`annotation/tau_success_calibration.html`.

- `gpt56.jsonl`: GPT-5.6 Codex blind judgments.
- `sonnet.jsonl`: `aws-claude-4-6-sonnet-fair` blind judgments through the
  configured OpenAI-compatible endpoint.
- `opus.jsonl`: Opus CLI blind judgments; the runtime resolved `opus` to
  `claude-opus-5`.
- `consensus.jsonl`: per-comparison votes joined with hidden generation roles.
- `summary.json`: panel agreement and the human-review queue.
- `checksums.sha256`: hashes of all machine-generated judgment artifacts.

Regenerate the aggregate after obtaining the three raw files:

```bash
python scripts/summarize_tau_success_judges.py
```

LLM agreement is weak supervision. It does not establish E1/E2 ground truth or
metric accuracy.
