#!/usr/bin/env python3
"""G3: does the derivation-content fingerprint construct transfer beyond math?

Domains: code (MBPP sanitized, correctness = tests pass) and csr (StrategyQA, yes/no).
Same regime as G2: Qwen3-8B non-thinking, temp 0.7 top_p 0.95, seed 0, K=8 x 50 problems.
Code twist: the AST is PARSER-GIVEN — the graph-reliability failure that killed G1/G2
topology cannot occur, so ast_wl is the live test of whether graph structure works when
extraction noise is zero. Code style rewrites are VERIFIED (must still pass tests).
csr is the stress test: no verifiable objects — raw_val expected to fail (reported).

  CUDA_VISIBLE_DEVICES=3 HF_HOME=/scratch/yirenl2/.cache/huggingface \
    .venv-vllm/bin/python scripts/34_domain_transfer.py --gen code --smoke
  CUDA_VISIBLE_DEVICES=3 HF_HOME=... .venv-vllm/bin/python scripts/34_domain_transfer.py --gen all
  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/34_domain_transfer.py --judge all     # est ~$5
  .venv/bin/python scripts/34_domain_transfer.py --paraphrase all
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/34_domain_transfer.py --encode all
  .venv/bin/python scripts/34_domain_transfer.py --eval code --eval csr
"""
import argparse, ast as pyast, collections, json, random, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path("runs/domain-transfer")
MODEL = "Qwen/Qwen3-8B"
REV = "b968826d9c46dd6066d109eabc6255188de91218"
K, NPROB, SEED, JUDGE_CAP = 8, 50, 0, 6000
JUDGE = "gpt-5.5-2026-04-23"

JUDGE_SYS = {
    "code": """You are given several Python solutions to the same programming task. Cluster them by ALGORITHMIC APPROACH.

Same approach = same underlying algorithm and data-structure strategy (e.g. both sort then scan; both use a hash map; both brute-force nested loops), even if they differ in variable names, comments, code style, iteration constructs, or minor refactors.
Different approach = the algorithmic idea itself differs (e.g. sorting-based vs hash-map vs brute force; regex vs manual parsing; math formula vs simulation; recursion implementing a genuinely different decomposition).

Output JSON:
{"methods": ["short description of approach 1", ...],
 "assignment": [m, m, ...]}   // 1-based approach index per solution, in the order given
Every solution must be assigned. Do not create a new approach for presentation differences.""",
    "csr": """You are given several reasoning traces answering the same yes/no question. Cluster them by REASONING ROUTE.

Same route = they rely on the same key facts and the same chain of inference (same decomposition of the question), even if worded differently, ordered differently, more or less verbose, or reaching different final answers via a slip.
Different route = the argument itself differs: different key facts invoked, different decomposition, a different bridging principle (e.g. arguing from dates vs from geography; from a definition vs from a typical example).

Output JSON:
{"methods": ["short description of route 1", ...],
 "assignment": [m, m, ...]}   // 1-based route index per solution, in the order given
Every trace must be assigned. Do not create a new route for presentation differences."""}

CODE_STYLES = {
    "R_renamed": "Rename every variable, argument, and helper to completely different but valid names. Change NOTHING else.",
    "S_restructured": "Restructure the code: reorder independent statements, replace loops with equivalent constructs (comprehension <-> for), inline or extract expressions. Keep the SAME algorithm and exact behavior.",
    "C_compressed": "Rewrite as compactly as possible (one-liners, comprehensions, chained expressions). Same algorithm, exact same behavior.",
    "V_verbose": "Rewrite verbosely: add docstrings, comments, type hints, intermediate named variables. Same algorithm, exact same behavior.",
}
CSR_STYLES = {
    "D_restructured": "Completely reorder the presentation (e.g. state the conclusion first, then justify; or reverse the argument order). Keep every fact and inference identical.",
    "K_compressed": "Rewrite as compactly as possible - telegraphic, minimal words. Keep every fact and inference identical.",
    "L_maximal": "Rewrite as verbosely as possible - expand every step, add discourse connectives. Keep every fact and inference identical, add no new facts.",
    "M_narrative": "Rewrite as a flowing first-person narrative ('I first considered...'). Keep every fact and inference identical.",
}
PARA_SYS = {"code": "Rewrite the Python solution below per the instruction. Output ONLY a ```python code block, no explanation. The rewritten code must behave identically.",
            "csr": "Rewrite the reasoning below per the instruction. Preserve the exact logical content and the final 'Answer: yes/no' line. Output only the rewrite."}


def load_problems(domain, nprob=NPROB):
    from datasets import load_dataset
    if domain == "code":
        d = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        items = [{"problem": r["prompt"], "tests": r["test_list"], "imports": r["test_imports"],
                  "gold": None} for r in d]
    else:
        d = load_dataset("ChilleD/StrategyQA", split="test")
        items = [{"problem": r["question"], "tests": None, "imports": None,
                  "gold": "yes" if r["answer"] else "no"} for r in d]
    random.Random(SEED).shuffle(items)
    return items[:nprob]


def user_prompt(domain, it):
    if domain == "code":
        t = "\n".join(it["tests"])
        return (f"{it['problem']}\n\nYour function must satisfy these tests:\n{t}\n\n"
                "Write the complete solution as a single ```python code block.")
    return (f"{it['problem']}\n\nReason step by step (briefly), then give your final answer "
            "on the last line as 'Answer: yes' or 'Answer: no'.")


def code_block(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text or "", re.S)
    return m[-1] if m else (text or "")


def run_tests(code, it, timeout=10):
    prog = "\n".join(it["imports"]) + "\n" + code + "\n" + "\n".join(it["tests"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog); fn = f.name
    try:
        r = subprocess.run([sys.executable, fn], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(fn).unlink(missing_ok=True)


def grade(domain, text, it):
    if domain == "code":
        return int(run_tests(code_block(text), it)), None
    m = re.findall(r"[Aa]nswer:\s*(yes|no)", text or "")
    ext = m[-1].lower() if m else None
    return int(ext == it["gold"]), ext


def gen(domain, smoke, nprob=NPROB):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    OUT.mkdir(exist_ok=True)
    items = load_problems(domain, nprob)
    k = K
    if smoke:
        items, k = items[:2], 2
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": user_prompt(domain, it)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for it in items]
    llm = LLM(model=MODEL, revision=REV, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=8192)
    jobs = [(q, s) for q in range(len(items)) for s in range(k)]
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024,
                         seed=SEED * 1000003 + q * K + s) for q, s in jobs]
    outs = llm.generate([prompts[q] for q, _ in jobs], sp)
    rows = []
    for (q, s), o in zip(jobs, outs):
        text = o.outputs[0].text
        ok, ext = grade(domain, text, items[q])
        rows.append({"qidx": q, "samp": s, "text": text, "extracted": ext, "correct": ok,
                     "problem": items[q]["problem"], "gold": items[q]["gold"]})
    fn = OUT / f"traces_{domain}{'_smoke' if smoke else ''}.jsonl"
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qidx"]].append(r["correct"])
    acc = np.mean([r["correct"] for r in rows])
    p8 = np.mean([int(any(v)) for v in byq.values()])
    print(f"[{domain}] {len(rows)} gens  acc {acc:.3f}  pass@{k} {p8:.3f}")


def traces(domain):
    return [json.loads(l) for l in open(OUT / f"traces_{domain}.jsonl")]


def judge_labels(domain, smoke):
    from openai import OpenAI
    client = OpenAI()
    byq = collections.defaultdict(dict)
    for r in traces(domain):
        byq[r["qidx"]][r["samp"]] = r
    fn_full = OUT / f"labels_{domain}.jsonl"
    done = {json.loads(l)["qidx"] for l in open(fn_full)} if fn_full.exists() else set()
    jobs = []
    for q in sorted(byq):
        if q in done:
            continue
        sols = [byq[q][s]["text"][:JUDGE_CAP] for s in sorted(byq[q])]
        jobs.append({"qidx": q, "problem": byq[q][0]["problem"], "sols": sols})
    if smoke:
        jobs = jobs[:2]
    est = sum(sum(len(t) for t in j["sols"]) // 3 + 1500 for j in jobs) * 2.5 / 1e6 + \
        len(jobs) * 1500 * 25 / 1e6
    print(f"[{domain}] judge jobs={len(jobs)}  est cost ~ ${est:.2f} ({JUDGE})")

    def one(j):
        body = f"TASK:\n{j['problem']}\n\n" + "\n\n".join(
            f"--- SOLUTION {i + 1} ---\n{t}" for i, t in enumerate(j["sols"]))
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=JUDGE, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": JUDGE_SYS[domain]},
                              {"role": "user", "content": body}],
                    max_completion_tokens=8000, reasoning_effort="low")
                out = json.loads(r.choices[0].message.content)
                asg = out.get("assignment", [])
                assert len(asg) == len(j["sols"])
                u = r.usage
                return {"qidx": j["qidx"], "assignment": asg, "methods": out.get("methods", []),
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    fn = OUT / f"labels_{domain}{'_smoke' if smoke else ''}.jsonl"
    with open(fn, "a" if fn.exists() and not smoke else "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nm = [len(set(r["assignment"])) for r in rows]
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"[{domain}] methods/prob mean {np.mean(nm):.2f}  %>=2m "
          f"{np.mean([n >= 2 for n in nm]):.2f}  %>=3m {np.mean([n >= 3 for n in nm]):.2f}  "
          f"cost ~ ${(ptok * 2.5 + ctok * 25.0) / 1e6:.2f}")


def pick_bases(domain):
    byq = collections.defaultdict(list)
    for r in traces(domain):
        byq[r["qidx"]].append(r)
    out = []
    for q in sorted(byq):
        pool = [r for r in byq[q] if r["correct"]] or byq[q]
        pool.sort(key=lambda r: abs(len(r["text"].split()) - (150 if domain == "code" else 120)))
        out.append(pool[0])
    return out


def paraphrase(domain, smoke):
    from openai import OpenAI
    client = OpenAI()
    styles = CODE_STYLES if domain == "code" else CSR_STYLES
    srcs = pick_bases(domain)
    items = load_problems(domain, max(r["qidx"] for r in srcs) + 1)
    if smoke:
        srcs = srcs[:2]
    jobs = [(r, sk, sv) for r in srcs for sk, sv in styles.items()]
    print(f"[{domain}] paraphrase jobs={len(jobs)} (gpt-4.1-mini, est ~ "
          f"${sum(len(r['text']) for r, _, _ in jobs) * 2 / 3 * 1.0 / 1e6:.2f})")

    def one(job):
        r, sk, sv = job
        src = code_block(r["text"]) if domain == "code" else r["text"]
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "system", "content": PARA_SYS[domain]},
                              {"role": "user", "content": f"INSTRUCTION: {sv}\n\nORIGINAL:\n{src}"}],
                    max_completion_tokens=2500, temperature=0.7)
                text = resp.choices[0].message.content or ""
                row = {"qidx": r["qidx"], "samp": r["samp"], "style": sk, "text": text}
                if domain == "code":
                    row["verified"] = int(run_tests(code_block(text), items[r["qidx"]]))
                return row
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    fn = OUT / f"paraphrases_{domain}{'_smoke' if smoke else ''}.jsonl"
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if domain == "code":
        v = [r["verified"] for r in rows]
        print(f"[{domain}] wrote {len(rows)}; verified (tests still pass) {sum(v)}/{len(v)}")
    else:
        print(f"[{domain}] wrote {len(rows)}")


def ids_texts(domain):
    tr = traces(domain)
    pa = [json.loads(l) for l in open(OUT / f"paraphrases_{domain}.jsonl")]
    ids = [f"n{r['qidx']}_{r['samp']}" for r in tr] + [f"p{r['qidx']}_{r['style']}" for r in pa]
    texts = [r["text"] for r in tr + pa]
    return ids, texts, pa


def encode(domain):
    import torch
    from transformers import AutoModel, AutoTokenizer
    ids, texts, _ = ids_texts(domain)
    if domain == "code":
        texts = [code_block(t) for t in texts]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B",
                                      dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=2048,
                    return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state[:, -1]
            embs.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    np.savez(OUT / f"emb_{domain}.npz", E=np.concatenate(embs), ids=np.array(ids))
    print(f"[{domain}] encoded {len(ids)} texts")


CLAIM_SYS = {
    "csr": """Extract the logical skeleton of the reasoning below as a list of atomic claims.
Each claim = one factual statement or inference the reasoning relies on, in canonical neutral wording (no narrative, no hedging), <= 15 words. 3-15 claims.
Output JSON: {"claims": ["...", ...]}""",
    "code": """Extract the algorithmic skeleton of the Python solution below as a list of atomic steps.
Each step = one operation the algorithm performs, in canonical neutral wording describing WHAT is computed, not how it is written (no variable names, no syntax, no style), <= 15 words. 3-15 steps.
Identical algorithms written differently must yield identical steps.
Output JSON: {"claims": ["...", ...]}"""}


def encode_l18(domain):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ids, texts, _ = ids_texts(domain)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV,
                                                 torch_dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    CH = 128
    chunks, trace = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 4):
            b = tok(texts[i:i + 4], padding=True, truncation=True, max_length=4096,
                    return_tensors="pt").to("cuda")
            hs = model(**b, output_hidden_states=True).hidden_states[18]
            for r in range(hs.shape[0]):
                v = hs[r][b["attention_mask"][r].bool()]
                cs = torch.stack([v[j:j + CH].mean(0) for j in range(0, v.shape[0], CH)])
                chunks.append(torch.nn.functional.normalize(cs, dim=-1).float().cpu().numpy())
                trace.append(torch.nn.functional.normalize(v.mean(0), dim=-1).float().cpu().numpy())
    E, off, k = np.concatenate(chunks), {}, 0
    for tid, c in zip(ids, chunks):
        off[tid] = (k, k + len(c)); k += len(c)
    np.savez(OUT / f"l18chunk_{domain}.npz", E=E, offsets=json.dumps(off))
    np.savez(OUT / f"l18trace_{domain}.npz", E=np.stack(trace), ids=np.array(ids))
    print(f"[{domain}] L18-encoded {len(ids)} texts, {len(E)} chunks")


def claims_extract(domain):
    from openai import OpenAI
    client = OpenAI()
    ids, texts, _ = ids_texts(domain)
    if domain == "code":  # traces carry prose, rewrites don't — compare on the code alone
        texts = [code_block(t) for t in texts]
    print(f"[{domain}] claims jobs={len(texts)} (gpt-4.1-mini, est ~ "
          f"${(sum(len(t) // 3 + 200 for t in texts) * 0.4 + len(texts) * 300 * 1.6) / 1e6:.2f})")

    def one(job):
        tid, text = job
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model="gpt-4.1-mini", response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": CLAIM_SYS[domain]},
                              {"role": "user", "content": text}],
                    max_completion_tokens=1200, temperature=0.0 + 0.3 * min(attempt, 1))
                cl = [str(c)[:150] for c in
                      json.loads(r.choices[0].message.content).get("claims", [])][:20]
                return {"id": tid, "claims": cl}
            except Exception:
                if attempt == 3:
                    return {"id": tid, "claims": []}
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, list(zip(ids, texts))))
    with open(OUT / f"claims_{domain}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import torch
    from transformers import AutoModel, AutoTokenizer
    flat = [(r["id"], c) for r in rows for c in r["claims"]]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B",
                                      dtype=torch.bfloat16).cuda().eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(flat), 128):
            b = tok([c for _, c in flat[i:i + 128]], padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state[:, -1]
            embs.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    E, off, k = (np.concatenate(embs) if flat else np.zeros((0, 1024))), {}, 0
    for r in rows:
        off[r["id"]] = (k, k + len(r["claims"])); k += len(r["claims"])
    np.savez(OUT / f"claimemb_{domain}.npz", E=E, offsets=json.dumps(off))
    print(f"[{domain}] claims: mean {np.mean([len(r['claims']) for r in rows]):.1f}/trace")


TRACER = r'''
import sys, json
vals, nl = set(), 0
def tr(frame, event, arg):
    global nl
    if event == "line" and frame.f_code.co_filename == "<prog>":
        nl += 1
        for v in list(frame.f_locals.values()):
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and 2 < abs(v) < 1e12:
                vals.add(str(v if isinstance(v, int) else round(v, 6)))
            elif isinstance(v, str) and 3 <= len(v) <= 50:
                vals.add("s:" + v)
    return tr
src = open(sys.argv[1]).read()
sys.settrace(tr)
try:
    exec(compile(src, "<prog>", "exec"), {})
except Exception:
    pass
sys.settrace(None)
print(json.dumps({"vals": sorted(vals)[:3000], "nlines": nl}))
'''


def behave(domain):
    # behavioral fingerprint: values materialized at runtime on the SHARED tests —
    # invariant to any semantics-preserving rewrite by construction
    assert domain == "code"
    ids, texts, _ = ids_texts(domain)
    items = load_problems(domain, max(int(i[1:].split("_")[0]) for i in ids) + 1)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(TRACER); tracer_fn = f.name

    def one(job):
        tid, text = job
        q = int(tid[1:].split("_")[0])
        it = items[q]
        prog = "\n".join(it["imports"]) + "\n" + code_block(text) + "\n" + "\n".join(it["tests"])
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog); fn = f.name
        try:
            r = subprocess.run([sys.executable, tracer_fn, fn], capture_output=True,
                               timeout=20, text=True)
            out = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
        except Exception:
            out = {}
        finally:
            Path(fn).unlink(missing_ok=True)
        return {"id": tid, "vals": out.get("vals", []), "nlines": out.get("nlines", 0)}

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(one, list(zip(ids, texts))))
    Path(tracer_fn).unlink(missing_ok=True)
    with open(OUT / "behave_code.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nv = [len(r["vals"]) for r in rows]
    print(f"[code] behavioral traces: {len(rows)}, vals/prog mean {np.mean(nv):.1f} "
          f"empty {np.mean([n == 0 for n in nv]):.2f}")


STOP = set("the a an is are was were be been being to of in on for with as by at from that "
           "this these those it its they their there then than and or not no yes but if so "
           "because since while would could should can may might will do does did done has "
           "have had having he she his her we you i one two also very more most much many "
           "some any all both each which what when where who whom how why answer question "
           "step first second therefore thus hence finally conclusion".split())


def word_set(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) >= 4 and w not in STOP}


def raw_vals(text, minv=2):
    out = set()
    for m in re.findall(r"-?\d+(?:/\d+)?", text or ""):
        try:
            v = Fraction(m)
        except Exception:
            continue
        if abs(v) > minv:
            out.add(v)
    return out


def ast_feats(code):
    # canonical AST: node-type labels only (identifier-free), WL h=2; + calls to
    # externally-defined names; + numeric literals
    try:
        tree = pyast.parse(code)
    except SyntaxError:
        return None
    defined = {n.name for n in pyast.walk(tree)
               if isinstance(n, (pyast.FunctionDef, pyast.AsyncFunctionDef))}
    children = collections.defaultdict(list)
    labels, nodes = [], []
    for i, n in enumerate(pyast.walk(tree)):
        nodes.append(n); labels.append(type(n).__name__)
    idx = {id(n): i for i, n in enumerate(nodes)}
    for n in nodes:
        for c in pyast.iter_child_nodes(n):
            children[idx[id(n)]].append(idx[id(c)])
    feats = collections.Counter(labels)
    lab = list(labels)
    for _ in range(2):
        lab = [lab[i] + "(" + ",".join(sorted(lab[c] for c in children[i])) + ")"
               for i in range(len(lab))]
        feats.update(lab)
    calls = set()
    for n in pyast.walk(tree):
        if isinstance(n, pyast.Call):
            f = n.func
            name = f.id if isinstance(f, pyast.Name) else \
                f.attr if isinstance(f, pyast.Attribute) else None
            if name and name not in defined:
                calls.add(name)
    nums = {n.value for n in pyast.walk(tree)
            if isinstance(n, pyast.Constant) and isinstance(n.value, (int, float))
            and abs(n.value) > 2}
    types = collections.Counter(labels)
    return {"wl": feats, "types": types, "calls": calls, "nums": nums}


def cos_counter(ca, cb):
    if not ca or not cb:
        return float("nan")
    num = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys())
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return float(1 - num / (na * nb + 1e-9))


def d_jac(A, B):
    if not A or not B:
        return float("nan")
    return 1 - len(A & B) / len(A | B)


def evaluate(domain):
    from sklearn.metrics import roc_auc_score
    ids, texts, pa = ids_texts(domain)
    txt = dict(zip(ids, texts))
    lab = {r["qidx"]: r["assignment"] for r in
           (json.loads(l) for l in open(OUT / f"labels_{domain}.jsonl"))}
    tr = traces(domain)
    ext = {f"n{r['qidx']}_{r['samp']}": (r["extracted"] if domain == "csr" else r["correct"])
           for r in tr}
    styles = list((CODE_STYLES if domain == "code" else CSR_STYLES))
    verified = {(r["qidx"], r["style"]) for r in pa
                if domain == "csr" or r.get("verified")}
    diff, same = [], []
    for q, a in lab.items():
        for i, j in combinations(range(K), 2):
            (diff if a[i] != a[j] else same).append((f"n{q}_{i}", f"n{q}_{j}"))

    if domain == "code":
        F = {t: ast_feats(code_block(txt[t])) for t in ids}
        m = {"ast_wl": lambda a, b: cos_counter(F[a] and F[a]["wl"], F[b] and F[b]["wl"]),
             "ast_types": lambda a, b: cos_counter(F[a] and F[a]["types"], F[b] and F[b]["types"]),
             "call_set": lambda a, b: d_jac(F[a] and F[a]["calls"], F[b] and F[b]["calls"]),
             "num_lit": lambda a, b: d_jac(F[a] and F[a]["nums"], F[b] and F[b]["nums"])}
        if (OUT / "behave_code.jsonl").exists():
            brows = [json.loads(l) for l in open(OUT / "behave_code.jsonl")]
            B = {r["id"]: set(r["vals"]) for r in brows}
            NL = {r["id"]: r.get("nlines", 0) for r in brows}
            m["behav_val"] = lambda a, b: d_jac(B.get(a), B.get(b))
            m["behav_cplx"] = lambda a, b: (
                abs(np.log(NL[a] / NL[b])) if NL.get(a, 0) > 0 and NL.get(b, 0) > 0
                else float("nan"))
    else:
        W = {t: word_set(txt[t]) for t in ids}
        V = {t: raw_vals(txt[t]) for t in ids}
        m = {"word_set": lambda a, b: d_jac(W[a], W[b]),
             "raw_val": lambda a, b: d_jac(V[a], V[b])}
    z = np.load(OUT / f"emb_{domain}.npz")
    E = {t: e for t, e in zip(z["ids"], z["E"])}
    m["emb"] = lambda a, b: float(1 - E[a] @ E[b])

    import importlib.util as _iu
    _spec = _iu.spec_from_file_location("_s6", "scripts/23_logdist_seqot.py")
    _s6 = _iu.module_from_spec(_spec); _spec.loader.exec_module(_s6)
    TOK = {t: set(re.findall(r"\w+", (txt[t] or "").lower())) for t in ids}
    m["token_jac"] = lambda a, b: d_jac(TOK[a], TOK[b])
    m["ans_diff"] = lambda a, b: float(ext.get(a) != ext.get(b))
    if (OUT / f"l18trace_{domain}.npz").exists():
        import torch
        zt = np.load(OUT / f"l18trace_{domain}.npz")
        T = {t: e for t, e in zip(zt["ids"], zt["E"])}
        zc = np.load(OUT / f"l18chunk_{domain}.npz")
        EC = zc["E"]
        coff = {k: tuple(v) for k, v in json.loads(str(zc["offsets"])).items()}
        m["chunk_cham"] = lambda a, b: _s6.d_chamfer(EC[slice(*coff[a])], EC[slice(*coff[b])])

        def head(pt):
            W = torch.nn.Linear(4096, 256, bias=False)
            W.load_state_dict(torch.load(pt, map_location="cpu"))
            return W

        def apply_head(W, X):
            import torch as th
            with th.no_grad():
                return th.nn.functional.normalize(
                    W(th.tensor(np.asarray(X)).float()), dim=-1).numpy()

        for tag, d15, d4 in (("inreg", "runs/regime-probe/head_l15_inreg.pt",
                              "runs/regime-probe/head_l4_inreg.pt"),
                             ("v2", "runs/logdist-testbed/head_l15_l4_L18_v2.pt",
                              "runs/logdist-testbed/head_l4_L18_v2.pt")):
            F15 = apply_head(head(d15), np.stack([T[t] for t in ids]))
            F15 = {t: f for t, f in zip(ids, F15)}
            FC = apply_head(head(d4), EC)
            m[f"l15_{tag}"] = (lambda a, b, F=F15: float(1 - F[a] @ F[b]))
            m[f"l4_{tag}"] = (lambda a, b, F=FC:
                              _s6.d_chamfer(F[slice(*coff[a])], F[slice(*coff[b])]))
        for tag in ("inreg", "v2"):
            hfs = [(m[f"l15_{tag}"], float(np.nanstd([m[f"l15_{tag}"](a, b) for a, b in diff]))),
                   (m[f"l4_{tag}"], float(np.nanstd([m[f"l4_{tag}"](a, b) for a, b in diff])))]
            m[f"ENSh_{tag}"] = (lambda a, b, hfs=hfs:
                                float(np.nanmean([f(a, b) / s for f, s in hfs])))
    if (OUT / f"claimemb_{domain}.npz").exists():
        zl = np.load(OUT / f"claimemb_{domain}.npz")
        CE = zl["E"]
        cloff = {k: tuple(v) for k, v in json.loads(str(zl["offsets"])).items()}
        m["claim_cham"] = lambda a, b: (
            _s6.d_chamfer(CE[slice(*cloff[a])], CE[slice(*cloff[b])])
            if cloff.get(a, (0, 0))[1] > cloff.get(a, (0, 0))[0]
            and cloff.get(b, (0, 0))[1] > cloff.get(b, (0, 0))[0] else float("nan"))
    fs = None
    ks = ["ast_wl", "call_set", "num_lit"] if domain == "code" else ["word_set", "emb"]
    fs = [(m[k], float(np.nanstd([m[k](a, b) for a, b in diff]))) for k in ks]
    m["ENS"] = lambda a, b: float(np.nanmean([f(a, b) / s for f, s in fs]))

    def ansmatch(p):
        return ext.get(p[0]) is not None and ext.get(p[0]) == ext.get(p[1])
    diff_m = [p for p in diff if ansmatch(p)]; same_m = [p for p in same if ansmatch(p)]
    base = {r["qidx"]: r["samp"] for r in pa}
    # packs: problems with >=3 judge methods; pack size 3 (methods rarer outside math)
    packs = {}
    for q, a in lab.items():
        rep = {}
        for s in range(K):
            rep.setdefault(a[s], s)
        if len(rep) >= 3:
            packs[q] = [f"n{q}_{s}" for s in sorted(rep.values())[:3]]
    packq = [q for q in sorted(packs)
             if sum((q, st) in verified for st in styles) >= 3]
    s6 = __import__("importlib.util", fromlist=["x"])
    import importlib.util as iu
    spec = iu.spec_from_file_location("s6", "scripts/23_logdist_seqot.py")
    s6 = iu.module_from_spec(spec); spec.loader.exec_module(s6)
    ctrl = "corr-matched" if domain == "code" else "ans-matched"
    print(f"\n[{domain}] {len(diff)} diff / {len(same)} same pairs "
          f"({ctrl}: {len(diff_m)}/{len(same_m)}); {len(packq)} packs (size 3)")
    print(f"{'metric':10s} {'AUC_D':>6s} {'ctrl':>6s} {'ALLst':>6s}   "
          f"{'V_sty':>5s} {'V_met':>5s}   R_set [95% CI]  hack%")
    rng = np.random.default_rng(20260819)
    res = {}
    for mn, f in m.items():
        dd = [x for x in (f(a, b) for a, b in diff) if not np.isnan(x)]
        ds = [x for x in (f(a, b) for a, b in same) if not np.isnan(x)]
        aucd = roc_auc_score([0] * len(ds) + [1] * len(dd), ds + dd)
        ddm = [x for x in (f(a, b) for a, b in diff_m) if not np.isnan(x)]
        dsm = [x for x in (f(a, b) for a, b in same_m) if not np.isnan(x)]
        aucm = roc_auc_score([0] * len(dsm) + [1] * len(ddm), dsm + ddm) \
            if ddm and dsm else float("nan")
        allpos = []
        for st in styles:
            v = [f(f"n{q}_{base[q]}", f"p{q}_{st}") for q in sorted(lab)
                 if (q, st) in verified and f"p{q}_{st}" in txt]
            allpos += [x for x in v if not np.isnan(x)]
        aucg = roc_auc_score([0] * len(allpos) + [1] * len(dd), allpos + dd)
        tau = abs(np.nanmean([f(a, b) for a, b in diff])) or 1.0
        vs, vm = [], []
        for q in packq:
            spack = [f"p{q}_{st}" for st in styles if (q, st) in verified][:3]
            for pack, acc in zip((spack, packs[q]), (vs, vm)):
                D = np.zeros((3, 3))
                for i, j in combinations(range(3), 2):
                    D[i, j] = D[j, i] = np.nan_to_num(f(pack[i], pack[j]), nan=tau)
                acc.append(s6.vendi(D, tau))
        vs, vm = np.array(vs), np.array(vm)
        r = (vm.mean() - 1) / max(vs.mean() - 1, 1e-9) if len(vs) else float("nan")
        if len(vs):
            bs = [(vm[i].mean() - 1) / max(vs[i].mean() - 1, 1e-9)
                  for i in rng.integers(0, len(vs), (2000, len(vs)))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        res[mn] = {"auc_d": aucd, "auc_ctrl": aucm, "auc_allst": aucg, "R": float(r),
                   "ci": [float(lo), float(hi)],
                   "V_style": float(vs.mean()) if len(vs) else None,
                   "V_meth": float(vm.mean()) if len(vm) else None}
        print(f"{mn:10s} {aucd:6.3f} {aucm:6.3f} {aucg:6.3f}   "
              f"{vs.mean() if len(vs) else float('nan'):5.2f} "
              f"{vm.mean() if len(vm) else float('nan'):5.2f}   "
              f"{r:5.2f} [{lo:4.2f},{hi:4.2f}]  {100 / max(r, 1e-9):3.0f}%")
    (OUT / f"g3_eval_{domain}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    for f in ("gen", "judge", "paraphrase", "encode", "eval", "behave", "l18", "claims"):
        ap.add_argument(f"--{f}", action="append", default=[],
                        choices=["code", "csr", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nprob", type=int, default=NPROB)
    a = ap.parse_args()

    def doms(v):
        return ["code", "csr"] if "all" in v else v
    OUT.mkdir(exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "seed": SEED, "model": MODEL, "rev": REV,
               "time": time.strftime("%FT%TZ", time.gmtime())},
              open(OUT / "manifest.json", "w"), indent=1)
    for d in doms(a.gen):
        gen(d, a.smoke, a.nprob)
    for d in doms(a.judge):
        judge_labels(d, a.smoke)
    for d in doms(a.paraphrase):
        paraphrase(d, a.smoke)
    for d in doms(a.encode):
        encode(d)
    for d in doms(a.behave):
        behave(d)
    for d in doms(a.l18):
        encode_l18(d)
    for d in doms(a.claims):
        claims_extract(d)
    for d in doms(a.eval):
        evaluate(d)


if __name__ == "__main__":
    main()
