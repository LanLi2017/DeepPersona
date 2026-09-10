#!/usr/bin/env python3
"""Reuse frozen74 statistics for17 FIRST-block-reward tasks; preserve74 unchanged."""
import argparse
import io
from pathlib import Path
import tokenize
import types

HELPER = Path('scripts/74_code_regrouping_analysis.py')
OUT = Path('runs/swe-diversity-selection/code-regrouping-first-block')


def implementation():
    source = HELPER.read_text()
    # Change only the seven literal task-count constants, never statistics or gates.
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    count = sum(t.type == tokenize.NUMBER and t.string == '22' for t in tokens)
    assert count == 7, 'Unexpected frozen74 source layout'
    tokens = [t._replace(string='17') if t.type == tokenize.NUMBER and t.string == '22' else t
              for t in tokens]
    source = tokenize.untokenize(tokens)
    replacements = {
        '22-task identical-multiset intervention': '17-task FIRST-block-reward identical-multiset intervention',
        "OUT = Path('runs/swe-diversity-selection/code-regrouping-interaction-v2')":
            "OUT = Path('runs/swe-diversity-selection/code-regrouping-first-block')",
        "NOTE = Path('docs/swe_regrouping_analysis_protocol.md')":
            "NOTE = Path('docs/swe_first_block_regrouping_analysis_protocol.md')",
        "Path('scripts/72_code_regrouping_interaction_v2.py')":
            "Path('scripts/76_code_regrouping_first_block.py')",
        "sources = [Path(__file__), NOTE,":
            "sources = [Path(__file__), Path('scripts/74_code_regrouping_analysis.py'), "
            "Path('docs/swe_regrouping_analysis_protocol.md'), "
            "Path('runs/swe-diversity-selection/code-regrouping-interaction-v2/analysis_protocol.json'), "
            "Path('runs/swe-diversity-selection/code-extraction-audit/protocol.json'), "
            "Path('runs/swe-diversity-selection/code-extraction-audit/completed_manifest.json'), NOTE,",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType('first_block_regrouping_analysis')
    module.__file__ = __file__
    exec(compile(source, str(HELPER) + ':first-block-task-count-wrapper', 'exec'), module.__dict__)
    return module


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'analyze', 'self-test'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    analysis = implementation()
    if args.stage == 'freeze':
        analysis.freeze(args.out)
    elif args.stage == 'analyze':
        analysis.analyze(args.out)
    else:
        analysis.self_test()
