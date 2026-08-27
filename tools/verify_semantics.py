# NpuEmbeddings -- does the embedding MEAN anything? A near/far ordering gate.
# SPDX-License-Identifier: Apache-2.0
#
# Every other accuracy gate in this repo is RELATIVE: it compares our vectors
# against sentence-transformers on the same texts and asks whether the numbers
# agree (`tools/verify_embed_e2e.py`, `--bench`'s golden gate, the MTEB run).
# Those are the right instruments for "is the datapath correct", and they share
# one blind spot -- **they pass when both sides are wrong the same way.**
#
#   * verify_embed_e2e prepends the SAME task prefix to the reference that the
#     runtime applies, so a wrongly-chosen prompt agrees with itself.
#   * A container whose `arch` was read wrong returns vectors that are
#     correctly shaped, correctly normed and bit-for-bit deterministic
#     (CLAUDE.md's own words about why `arch` is a whitelist).
#   * tasks/0110: a silently truncated text returned a perfectly ordinary
#     vector, and two documents sharing a preamble returned BYTE-IDENTICAL
#     ones.
#
# This gate needs no reference. It asks the only question a user actually has:
# do sentences about the same thing come out closer than sentences about
# different things? And it answers with ORDER, never with a tolerance -- there
# is no epsilon anywhere in this file, because "how close is close" is a
# property of each model's embedding space and comparing it across six models
# would be comparing scales, not correctness.
#
# STDLIB ONLY, deliberately. No numpy, no torch, no .venv-ref. 36 vectors of at
# most 1024 floats is 630 dot products -- milliseconds in pure Python -- and
# paying that buys the ability to run this against a COLD dist zip on a machine
# with nothing installed but CPython and the exe. That is exactly the situation
# in which "the vectors are meaningless" is most likely and least likely to be
# noticed.
#
# Usage:
#   python tools\verify_semantics.py                      # all built-in models
#   python tools\verify_semantics.py --models bge-base-en-v1.5
#   python tools\verify_semantics.py --exe dist\npuembeddings-0.4.0\npuembed.exe

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent

# The six built-ins, in catalogue order. Extra containers in models/ are NOT
# swept by default: `npuembeddings add` lets a user install finetunes, and a
# finetune's semantics are the user's business, not this gate's.
BUILTIN = [
    "all-MiniLM-L6-v2",
    "bge-small-en-v1.5",
    "bge-base-en-v1.5",
    "bge-large-en-v1.5",
    "nomic-embed-text-v1.5",
    "embeddinggemma-300m",
]

# tools/npue.py's header, re-declared rather than imported: npue.py needs numpy
# and the whole point of this file is that it does not.
_HEADER = "<4sIII QQQQ 16s"
_HEADER_SIZE = 64
assert struct.calcsize(_HEADER) == _HEADER_SIZE


def container_config(path: Path) -> dict:
    """The .npue config dict, read with nothing but `struct` and `json`."""
    with open(path, "rb") as f:
        magic, version, arch, _flags, jo, jl, _do, _dl, _rsv = struct.unpack(
            _HEADER, f.read(_HEADER_SIZE))
        if magic != b"NPUE":
            raise SystemExit(f"{path}: not a .npue file (magic {magic!r})")
        f.seek(jo)
        directory = json.loads(f.read(jl).decode("utf-8"))
    cfg = dict(directory["config"])
    cfg["_arch_field"] = arch
    cfg["_version"] = version
    return cfg


def read_f32(path: Path, rows: int, cols: int) -> list[list[float]]:
    """Read the [rows, cols] fp32 block `--embed` wrote.

    CHECK the byte count against the width read from the container; do not
    infer the width from the file. An inferred width reshapes a truncated file
    into a plausible array -- the same fail-open shape as trap 8, and
    verify_embed_e2e.py's own comment says so about its own reader.
    """
    raw = path.read_bytes()
    want = rows * cols * 4
    if len(raw) != want:
        raise SystemExit(f"{path}: {len(raw)} bytes, expected {want} "
                         f"({rows} texts x {cols} fp32)")
    a = array("f")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    return [list(a[i * cols:(i + 1) * cols]) for i in range(rows)]


def unit(v: list[float]) -> list[float]:
    """Normalise here rather than trusting the container's `l2_normalize`.

    T33 found that field written and never read. Cosine needs unit vectors and
    this costs 36 square roots, so the cheap thing is to stop depending on it.
    """
    n = math.sqrt(sum(x * x for x in v))
    if not (n > 0.0) or not math.isfinite(n):
        raise SystemExit("a returned embedding has zero or non-finite norm")
    return [x / n for x in v]


def cosine_matrix(vs: list[list[float]]) -> list[list[float]]:
    n = len(vs)
    m = [[1.0] * n for _ in range(n)]
    for i in range(n):
        vi = vs[i]
        for j in range(i + 1, n):
            d = sum(a * b for a, b in zip(vi, vs[j]))
            m[i][j] = m[j][i] = d
    return m


def word_overlap(a: str, b: str) -> float:
    """Jaccard over lowercased alphanumeric words.

    Used ONLY to rank far pairs for reporting -- the adversarial cases are
    derived from the corpus rather than hand-picked, so they cannot be quietly
    chosen to pass.
    """
    def toks(s):
        out, cur = set(), []
        for ch in s.lower():
            if ch.isalnum():
                cur.append(ch)
            elif cur:
                out.add("".join(cur))
                cur = []
        if cur:
            out.add("".join(cur))
        return out
    x, y = toks(a), toks(b)
    return len(x & y) / len(x | y) if (x | y) else 0.0


def embed(exe: Path, root: Path, model: str, texts: list[str], hidden: int,
          prefix: str | None, threads: int,
          artifacts: str | None) -> tuple[list[list[float]], dict]:
    """Returns the unit vectors AND what the runtime said it used.

    SCRAPED FROM THE RUNTIME'S OWN STATUS LINE, never restated from what this
    file asked for -- the same discipline tasks/0105 had to adopt for
    `datapath_reported` in the release harness. It matters here because the
    catalogue is per-model since tasks/0104: five models on bfp16-emulated
    MMAC, bge-small on plain bf16. A gate that validates vectors without
    recording which datapath produced them is one artifact-set mix-up away
    from certifying the wrong build.
    """
    with tempfile.TemporaryDirectory(prefix="semcheck_") as td:
        d = Path(td)
        (d / "in.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
        # THE SHIPPED CLI FORM, not the older `<root> --model X --embed in out`
        # one. They are the same binary and they do NOT select the same design
        # set: the legacy form falls back to `runtime/artifacts`, a per-op set
        # that predates the unified xclbin, and dies on any model wider than
        # 384 ("staged buffer for argument 1 is 3538944 bytes, design allows
        # 884736" on bge-base; a b_layout_hash refusal on bge-large). It fails
        # loudly both times -- the guards work -- but it means every harness
        # using that form has to pass --artifacts by hand, which is why nobody
        # noticed. `embed <model> <in> <out> --root <dir>` auto-picks
        # correctly on all six, in the dev tree and in a dist. tasks/0121 sec 4.
        #
        # Driving the shipped form is also the point: it is what `embed.cmd`
        # in a release zip runs, so this gate tests the thing a user gets.
        cmd = [str(exe), "embed", model, str(d / "in.txt"), str(d / "out.f32"),
               "--root", str(root), "--threads", str(threads)]
        if artifacts:
            cmd += ["--artifacts", artifacts]
        # Pass --prefix ONLY when the container has a prompts table. The
        # runtime refuses the flag on a model without one, and refuses to run
        # WITHOUT it on a model that has one (tasks/0118, resolve_prefix()) --
        # so this branch is not politeness, it is the contract.
        if prefix is not None:
            cmd += ["--prefix", prefix]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            sys.stdout.write(r.stdout or "")
            sys.stderr.write(r.stderr or "")
            raise SystemExit(f"{model}: npuembeddings embed failed "
                             f"({r.returncode})")
        said = {}
        for line in (r.stdout or "").splitlines():
            m = re.match(r"\s{2}(datapath|designs|toolchain|shape)\s{2,}(.+)",
                         line)
            if m and m.group(1) not in said:
                said[m.group(1)] = m.group(2).strip()
        vs = [unit(v) for v in read_f32(d / "out.f32", len(texts), hidden)]
        return vs, said


def _score(vs, ids, clus, is_probe, texts):
    """Everything downstream of the embeddings. Called twice: once on
    the real vectors and once on a rotated copy (the negative
    control), so the gate has to demonstrate it can fail."""
    n = len(vs)
    cos = cosine_matrix(vs)
    G = [i for i in range(n) if not is_probe[i]]
    # ---- THE GATE: pure ordering over the .a/.b pairs ---------------------
    # For every gate sentence, its own cluster partner must be the nearest of
    # the other 23. No threshold, no tolerance, and no dependence on the size
    # of the corpus -- add a cluster and every existing sentence's test is
    # unchanged in kind.
    rank_fail = []
    for i in G:
        order = sorted((j for j in G if j != i), key=lambda j: -cos[i][j])
        if clus[order[0]] != clus[i]:
            partner = next(j for j in G if j != i and clus[j] == clus[i])
            rank_fail.append({
                "id": ids[i], "nearest": ids[order[0]],
                "nearest_cos": cos[i][order[0]],
                "partner": ids[partner], "partner_cos": cos[i][partner],
                "partner_rank": 1 + order.index(partner)})

    near = sorted((cos[i][j], i, j) for a, i in enumerate(G)
                  for j in G[a + 1:] if clus[i] == clus[j])
    far = sorted((cos[i][j], i, j) for a, i in enumerate(G)
                 for j in G[a + 1:] if clus[i] != clus[j])

    # ---- The only number in the file, and it is not a semantic one --------
    # Two DIFFERENT topics coming back numerically identical means the encoder
    # collapsed, not that it disagrees about meaning. tasks/0110 is why this is
    # explicit: a truncated text returned a correctly shaped, correctly normed,
    # deterministic vector, and two documents sharing a preamble returned
    # BYTE-IDENTICAL ones. Under a pure ranking gate that shows up as arbitrary
    # tie-breaking, which is a confusing way to be told the encoder is broken.
    collapsed = [{"pair": [ids[i], ids[j]], "cos": c}
                 for c, i, j in far if c > 0.9999]

    # ---- THE PROBE: reported, never gated ---------------------------------
    probes = []
    for i in range(n):
        if not is_probe[i]:
            continue
        order = sorted(G, key=lambda j: -cos[i][j])
        best = next(k for k, j in enumerate(order) if clus[j] == clus[i])
        probes.append({"id": ids[i], "nearest": ids[order[0]],
                       "own_cluster_rank": best + 1,
                       "hit": clus[order[0]] == clus[i],
                       "cos_to_own": cos[i][order[best]]})
    hits = sum(1 for p in probes if p["hit"])

    # ---- Reported, deliberately NOT gated: global separation --------------
    # "the weakest near pair beats the strongest far pair" is the right
    # INTUITION and the wrong GATE: a min over 12 near pairs against a max over
    # 264 far ones is two extreme-value statistics, so it gets harder as the
    # corpus grows for unchanged model quality. verify_embed_e2e.py makes
    # exactly this argument about a fixed epsilon over 23 M pairs -- "testing
    # against a constant tests the corpus size". It is also not a correctness
    # question: two clusters can be legitimately adjacent without any model
    # being wrong. Reported on every run so a regression is visible; it just
    # does not decide pass/fail.
    worst_near, best_far = near[0], far[-1]
    adv = sorted(far, key=lambda r: -word_overlap(texts[r[1]], texts[r[2]]))[:3]

    return {
        "n_gate": len(G), "n_probe": len(probes),
        "n_near": len(near), "n_far": len(far),
        "gate_rank_failures": rank_fail,
        "collapsed_pairs": collapsed,
        "worst_near": {"cos": worst_near[0],
                       "pair": [ids[worst_near[1]], ids[worst_near[2]]]},
        "best_far": {"cos": best_far[0],
                     "pair": [ids[best_far[1]], ids[best_far[2]]]},
        "margin": worst_near[0] - best_far[0],
        "separated": worst_near[0] > best_far[0],
        "near_mean": sum(c for c, _, _ in near) / len(near),
        "far_mean": sum(c for c, _, _ in far) / len(far),
        "probe_hits": hits, "probe_rate": hits / len(probes),
        "probes": probes,
        "adversarial_far": [
            {"pair": [ids[i], ids[j]], "cos": c,
             "word_overlap": word_overlap(texts[i], texts[j])}
            for c, i, j in adv],
        "pass": (not rank_fail) and (not collapsed),
    }


def check_model(model: str, corpus: dict, exe: Path, root: Path, args) -> dict:
    """Embed the whole corpus once, then score the GATE and the PROBE apart.

    THE SPLIT IS THE DESIGN DECISION OF THIS FILE, and it was forced by
    measurement, not taste (tasks/0121 sec 3).

      GATE  = the .a/.b sentences, two per cluster: a lexical paraphrase pair.
              "fishing in the river" against "fishing in the lake" against "the
              office chair from IKEA". This is the property that was actually
              asked for, and all six models hold it comfortably.

      PROBE = the .c sentences, one per cluster: same topic, deliberately
              almost no shared words. Reported, never gated.

    The probe started life as a third gated sentence and had to be demoted.
    Its failures were consistently NOT the runtime being wrong -- they were
    "is a hedge related to a rose", "is a timing belt like an aircraft fault",
    "is salmon fishing or food". Those are questions about a checkpoint's world
    knowledge, which is what MTEB is for, and a gate that fires on them would
    cry wolf until somebody switched it off. Keeping the signal and dropping
    the veto keeps the interesting half without the false alarms.

    The probe is also scored AGAINST THE GATE SENTENCES ONLY, never against
    other probes. Left in the same pool the twelve probes formed an accidental
    thirteenth cluster -- four of the six models made `car.c` and `med.c` each
    other's nearest neighbours, because both were first-person errand stories.
    The models were right; the corpus was measuring narrative voice.
    """
    # PER-MODEL LANGUAGE SUBSET (v6, tasks/0137). Sentences may carry a
    # "lang" field; missing means English. Non-English clusters are scored
    # ONLY for models the corpus names in `multilingual_models` -- the six
    # English shipped models never claimed to separate paraphrases in
    # languages they were not trained for, so gating them there would
    # measure the corpus, not the runtime (tasks/0121 sec 3's trap, one
    # language over). The six therefore see EXACTLY the pre-v6 36 sentences.
    ml_models = (corpus.get("multilingual_models") or {}).get("models", [])
    sents = [s for s in corpus["sentences"]
             if s.get("lang", "en") == "en" or model in ml_models]
    texts = [s["text"] for s in sents]
    ids = [s["id"] for s in sents]
    clus = [s["cluster"] for s in sents]
    is_probe = [s.get("kind") == "semantic" for s in sents]
    n = len(sents)

    cfg = container_config(root / "models" / f"{model}.npue")
    hidden = int(cfg["hidden"])
    prompts = cfg.get("prompts") or {}

    prefix = None
    if prompts:
        want = corpus["prompts"].get(model)
        if want is None:
            raise SystemExit(
                f"{model} has a prompts table {sorted(prompts)} but "
                f"semantic_corpus.json names no prompt for it. Refusing to "
                f"pick one -- a wrongly-prompted embedding is correctly shaped "
                f"and correctly normed, so this gate could not tell.")
        if want not in prompts:
            raise SystemExit(
                f"{model}: corpus asks for prompt {want!r}, container carries "
                f"{sorted(prompts)}. The corpus and the container have drifted.")
        prefix = want

    vs, said = embed(exe, root, model, texts, hidden, prefix, args.threads,
                     args.artifacts)

    # NEGATIVE CONTROL, run on every invocation because it is free.
    #
    # A gate that has never failed is not known to work. tasks/0094 is this
    # project's own precedent: the golden gate was structurally blind to
    # cross-row corruption AND was comparing 4 of 128 rows, and neither was
    # visible from the fact that it passed. So the same scoring runs a second
    # time over the SAME embeddings with the vectors rotated by one -- every
    # sentence wearing its neighbour's vector. That must FAIL, and if it does
    # not, the gate is not measuring what it claims and this run is void
    # whatever the real result said.
    #
    # HOW to break it is not obvious, and the first two attempts did not.
    #
    #  * Rotating the whole list by one is nearly the identity here: the corpus
    #    is stored cluster-major in groups of three, so `fish.a` picks up
    #    `fish.b`'s vector and stays in its own cluster. It caught five models
    #    by luck and nomic PASSED THE CONTROL, which is how the flaw surfaced.
    #  * Permuting whole clusters does not work either, and cannot: the gate
    #    asks only "is my partner nearest", and relabelling clusters
    #    consistently preserves exactly that. Any control has to break the
    #    PAIRING, not the labelling.
    #
    # Rotating within the GATE subset by one does break it, provably. Cluster k
    # holds gate slots 2k and 2k+1; after the rotation they carry the vectors
    # of k's .b and of k+1's .a -- different clusters, so never a near pair --
    # while k.b's true best match now sits in cluster k-1's slot. Every one of
    # the 24 must fail, on every model, by construction.
    #
    # Rotation, not a shuffle: no RNG, no seed, identical on every machine.
    G0 = [i for i in range(n) if not is_probe[i]]
    rotated = list(vs)
    for a, i in enumerate(G0):
        rotated[i] = vs[G0[(a + 1) % len(G0)]]
    real = _score(vs, ids, clus, is_probe, texts)
    control = _score(rotated, ids, clus, is_probe, texts)

    out = dict(real)
    out.update({
        "model": model, "hidden": hidden, "arch": cfg.get("_arch_field"),
        "prompt": prefix,
        "datapath_reported": said.get("datapath", "UNREPORTED"),
        "designs_reported": said.get("designs", "UNREPORTED"),
        "toolchain_reported": said.get("toolchain", "UNREPORTED"),
        "control_pass": control["pass"],
        "control_gate_failures": len(control["gate_rank_failures"]),
    })
    # The control MUST fail. If rotating every vector by one still puts every
    # paraphrase pair together, the scoring is not reading the vectors.
    out["pass"] = bool(real["pass"] and not control["pass"])
    out["control_ok"] = not control["pass"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=BUILTIN)
    ap.add_argument("--corpus", default=str(REPO / "tools" /
                                            "semantic_corpus.json"))
    ap.add_argument("--exe", default=None,
                    help="npuembeddings.exe -- a dev build or the one in a "
                         "dist zip (default: runtime/build)")
    ap.add_argument("--root", default=None,
                    help="the directory holding models/ and artifacts_* "
                         "(default: the repo)")
    ap.add_argument("--artifacts", default=None,
                    help="override the design set; by default the runtime "
                         "picks, which exercises pick_artifacts() too")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else REPO
    exe = (Path(args.exe).resolve() if args.exe
           else REPO / "runtime" / "build" / "npuembeddings.exe")
    if not exe.exists():
        raise SystemExit(f"no runtime at {exe} -- build it, or pass --exe")

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    sents = corpus["sentences"]
    shape = {}
    for s in sents:
        g, p = shape.setdefault(s["cluster"], [0, 0])
        shape[s["cluster"]] = ([g, p + 1] if s.get("kind") == "semantic"
                               else [g + 1, p])
    bad = {k: v for k, v in shape.items() if v != [2, 1]}
    if bad:
        raise SystemExit(f"every cluster needs exactly 2 gate sentences and 1 "
                         f"probe (.a/.b/.c); these do not: {bad}")
    ncl = len(shape)

    n_ml = sum(1 for s in sents if s.get("lang", "en") != "en") // 3
    print(f"semantic gate -- {ncl} clusters of .a/.b/.c "
          f"({ncl - n_ml} English + {n_ml} lang-tagged; non-English clusters "
          f"score only for the corpus's multilingual_models)")
    print(f"exe {exe}")
    print("THE GATE IS ORDER: every gate sentence's own cluster partner must")
    print("be its NEAREST of the other gate sentences. No tolerance -- the one")
    print("number in the file catches a COLLAPSED encoder, not a semantic")
    print("disagreement. The probe row and the global margin are REPORTED and")
    print("never gated; the reasoning is in check_model()'s docstring.\n")

    results = []
    for m in args.models:
        if not (root / "models" / f"{m}.npue").exists():
            print(f"  {m:<24} SKIP -- no container")
            continue
        r = check_model(m, corpus, exe, root, args)
        results.append(r)
        print(f"  {m:<24} {'PASS' if r['pass'] else 'FAIL'}   "
              f"gate {r['n_gate'] - len(r['gate_rank_failures'])}"
              f"/{r['n_gate']}   "
              f"near {r['near_mean']:+.3f} vs far {r['far_mean']:+.3f}   "
              f"margin {r['margin']:+.3f}   "
              f"probe {r['probe_hits']}/{r['n_probe']}   "
              f"ctrl {'fails-as-required' if r['control_ok'] else 'DID NOT FAIL'}"
              + (f"   prompt {r['prompt']!r}" if r["prompt"] else ""))
        print(f"{'':>26}{r['datapath_reported']}"
              f"   [{r['designs_reported']}]")
        for f in r["gate_rank_failures"]:
            print(f"        GATE: {f['id']} is nearest {f['nearest']} "
                  f"({f['nearest_cos']:+.3f}); its partner {f['partner']} "
                  f"({f['partner_cos']:+.3f}) is only rank "
                  f"{f['partner_rank']}")
        for c in r["collapsed_pairs"]:
            print(f"        COLLAPSED: {c['pair']} are numerically identical "
                  f"({c['cos']:.6f}) but are different topics")

    if not results:
        raise SystemExit("no models checked")

    print(f"\n  {'model':<24}{'near min':>10}{'far max':>9}{'margin':>9}"
          f"{'probe':>8}")
    for r in results:
        print(f"  {r['model']:<24}{r['worst_near']['cos']:>10.3f}"
              f"{r['best_far']['cos']:>9.3f}{r['margin']:>9.3f}"
              f"{r['probe_hits']:>5}/{r['n_probe']}")
    print("\n  near min / far max are the extremes the GATE does not use: a")
    print("  negative margin means two unrelated clusters sit closer than the")
    print("  weakest paraphrase pair, which is a fact about English, not a")
    print("  regression. Watch it move, do not gate on it.")

    print("\n  probe misses (same topic, almost no shared words -- model")
    print("  quality, not runtime correctness):")
    any_miss = False
    for r in results:
        miss = [p for p in r["probes"] if not p["hit"]]
        if not miss:
            continue
        any_miss = True
        print(f"    {r['model']:<24} "
              + ", ".join(f"{p['id']}->{p['nearest']}(own rank "
                          f"{p['own_cluster_rank']})" for p in miss))
    if not any_miss:
        print("    none -- every probe found its own cluster first")

    print("\n  worst far pairs by shared words (derived, not hand-picked):")
    for r in results:
        a = r["adversarial_far"][0]
        print(f"    {r['model']:<24} {a['pair'][0]}/{a['pair'][1]} "
              f"overlap {a['word_overlap']:.2f}  cos {a['cos']:+.3f}  "
              f"(weakest near {r['worst_near']['cos']:+.3f})")

    ok = all(r["pass"] for r in results)
    out = Path(args.out) if args.out else (
        REPO / "tasks" / "0121-semantic-gate" / "semantic_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "behavioural gate, ordering only -- no tolerance",
        "corpus": os.path.basename(args.corpus),
        "corpus_version": corpus["version"],
        "exe": str(exe),
        "models": results, "pass": ok,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("PASS -- every model puts each paraphrase pair together" if ok
          else "FAIL -- see the GATE rows above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
