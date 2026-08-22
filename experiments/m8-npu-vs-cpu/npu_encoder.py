# NpuEmbeddings -- an MTEB-compatible encoder backed by the C++ NPU runtime.
# SPDX-License-Identifier: Apache-2.0
#
# THE BRIDGE, and why it is shaped like this
# ------------------------------------------
# MTEB needs `encode(list[str]) -> ndarray`. The C++ runtime needs a tokenizer
# it does not have (WordPiece is unwritten), and CLAUDE.md rule 5 forbids
# Python at runtime. Both constraints are satisfied by keeping the boundary
# where the project already puts it: **data crosses as FILES, never imports.**
#
#   text --[HF tokenizer, this file]--> emb_sum.f32 + masks
#        --[npuembed.exe --encode-file]--> out.f32 --> embeddings
#
# The tokenizer and the embedding lookup are a gather and three adds; they are
# not the datapath under test. Using HF's tokenizer for BOTH sides is what
# isolates the question the gate actually asks: does the NPU datapath preserve
# embedding QUALITY, not merely fidelity to fp32 on four fixed sentences.
#
# SEQUENCE LENGTH. The compiled designs are seq 64, so texts are truncated to
# 64 tokens. The CPU comparator must be configured identically
# (`max_seq_length = 64`) or the comparison is dishonest -- longer texts would
# give the CPU strictly more information. Absolute MTEB scores here are
# therefore "MiniLM at seq 64", below the published seq-256 numbers, and only
# the NPU-vs-CPU DIFFERENCE is the claim.
#
# Env: .venv-ref  (transformers, torch, mteb)
# Usage:
#   from npu_encoder import NpuEncoder
#   NpuEncoder().encode(["hello", "world"])

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
SEQ = 64
# Width comes from the checkpoint in __init__; 384 was a
# literal that bge-large's 1024 falsified.


class NpuEncoder:
    """Runs the C++ NPU runtime as a subprocess, one call per encode()."""

    def __init__(self, artifacts="artifacts_b128il", threads=24, pipeline=2,
                 exe=None, model_dir=None, verbose=False,
                 model="all-MiniLM-L6-v2", prompt=None):
        from transformers import AutoTokenizer
        from safetensors.numpy import load_file

        self.exe = Path(exe or REPO / "runtime" / "build" / "npuembed.exe")
        if not self.exe.exists():
            raise FileNotFoundError(f"{self.exe} -- build the runtime first")
        self.artifacts = artifacts
        # Which container the runtime should load. Pooling and depth live in
        # it, so the caller does not have to know them.
        self.model = model
        self.threads = threads
        self.pipeline = pipeline
        self.verbose = verbose

        md = Path(model_dir or REPO / "models" / model)
        self.tok = AutoTokenizer.from_pretrained(str(md))

        # The three embedding tables, straight from the checkpoint. The .npue
        # holds them too, but bf16-packed and pre-tiled for the GEMM path; the
        # embedding sum is host work in fp32 on both paths, so read the source.
        st = md / "model.safetensors"
        w = load_file(str(st)) if st.exists() else None
        if w is None:
            import torch
            sd = torch.load(md / "pytorch_model.bin", map_location="cpu")
            w = {k: v.numpy() for k, v in sd.items()}
        def pick(*names):
            for n in names:
                for k in w:
                    if k.endswith(n):
                        return np.asarray(w[k], dtype=np.float32)
            raise KeyError(names)
        self._meta = None
        self.word = pick("embeddings.word_embeddings.weight")
        self.hidden = int(self.word.shape[1])
        self.typ = pick("embeddings.token_type_embeddings.weight")

        # ARCH, PREFIX AND POSITIONS, read from the container rather than
        # assumed (tasks/0071). Two things here were literals that a second
        # architecture falsifies:
        #
        #  * `pos` -- arch=2 (nomic) has NO absolute position table at all; it
        #    uses RoPE, which the runtime applies inside Encoder::run(). Calling
        #    pick() for it would raise KeyError, so this bridge simply could not
        #    load nomic before.
        #  * the task PREFIX -- nomic requires one ("search_document: "), and
        #    MTEB is the only gate that can catch a wrong or missing one, since
        #    both sides of a 1-cos comparison would use the same wrong prefix
        #    and agree perfectly.
        sys.path.insert(0, str(REPO / "tools"))
        from npue import Reader                                    # noqa: E402
        with Reader(str(REPO / "models" / f"{model}.npue")) as npue:
            cfg = npue.config
        self.arch = cfg.get("arch", "bert_abs_gelu_postln")
        self.rope = cfg.get("position_embedding_type") == "rope"
        prompts = cfg.get("prompts") or {}
        name = prompt if prompt is not None else cfg.get("prompt_default")
        if prompts and name is not None and name not in prompts:
            raise SystemExit(f"--prompt {name!r} is not in this container's "
                             f"prompts table {sorted(prompts)}")
        self.prompt_name = name if prompts else None
        self.prefix = prompts.get(name, "") if prompts else ""

        self.pos = None if self.rope else pick(
            "embeddings.position_embeddings.weight")
        if self.verbose or self.prefix:
            print(f"[npu_encoder] {model}: arch={self.arch} "
                  f"positions={'rope (runtime-side)' if self.rope else 'absolute table'} "
                  f"prefix={self.prefix!r}"
                  f"{'' if self.prefix else ' (none)'}", file=sys.stderr)

    def _encode_texts(self, sentences):
        sentences = [s if isinstance(s, str) else str(s) for s in sentences]
        if not sentences:
            return np.zeros((0, self.hidden), dtype=np.float32)
        n = len(sentences)
        # The prefix is plain text before [CLS] (verified tasks/0068 sec 4), so
        # it is prepended BEFORE tokenization and eats into the SEQ budget --
        # 4 of the 62 usable slots for "search_document: ". That is the real
        # cost and it must be inside the truncation, not bolted on after it.
        if self.prefix:
            sentences = [self.prefix + s for s in sentences]
        enc = self.tok(sentences, padding="max_length", truncation=True,
                       max_length=SEQ, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        am = enc["attention_mask"].astype(np.float32)
        tt = enc.get("token_type_ids")
        tt = (np.zeros_like(ids) if tt is None else np.asarray(tt, np.int64))

        # word + position + token_type, exactly reference/encoder.py's embed()
        # -- except arch=2, which has no position table: RoPE is applied inside
        # the runtime, on Q and K, after the qkv GEMM.
        emb = self.word[ids] + self.typ[tt]
        if self.pos is not None:
            emb = emb + self.pos[:SEQ][None, :, :]
        add_mask = np.where(am > 0, np.float32(0.0), np.float32(-1.0e30))

        with tempfile.TemporaryDirectory(prefix="npuenc_") as td:
            d = Path(td)
            (d / "emb_sum.f32").write_bytes(
                np.ascontiguousarray(emb, np.float32).tobytes())
            (d / "add_mask.f32").write_bytes(
                np.ascontiguousarray(add_mask, np.float32).tobytes())
            (d / "attention_mask.f32").write_bytes(
                np.ascontiguousarray(am, np.float32).tobytes())

            cmd = [str(self.exe), "..", "--model", self.model,
                   "--artifacts", self.artifacts,
                   "--threads", str(self.threads), "--encode-file", str(d)]
            if self.pipeline > 1:
                cmd += ["--pipeline", str(self.pipeline)]
            r = subprocess.run(cmd, cwd=str(REPO / "runtime"),
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"npuembed failed ({r.returncode}):\n"
                                   f"{r.stdout}\n{r.stderr}")
            if self.verbose:
                print(r.stdout.strip().splitlines()[-1])
            out = np.frombuffer((d / "out.f32").read_bytes(),
                                dtype=np.float32)
        if out.size != n * self.hidden:
            raise RuntimeError(f"expected {n * self.hidden} floats, got {out.size}")
        return out.reshape(n, self.hidden).copy()

    @property
    def mteb_model_meta(self):
        if self._meta is None:
            from mteb.models.model_meta import ModelMeta
            self._meta = ModelMeta(
                # NAME THE MODEL BEING MEASURED. This was the literal
                # "NpuEmbeddings/all-MiniLM-L6-v2-npu" while `revision` was
                # already parameterised, so every MTEB result for bge-base,
                # bge-large or nomic would have been filed under MiniLM's name
                # -- a fail-open that mislabels the answer rather than breaking
                # (tasks/0071). Only MiniLM and bge-base had ever been run,
                # which is why it survived.
                name="NpuEmbeddings/" + self.model + "-npu",
                revision="npu-" + self.model + "-" + self.artifacts,
                release_date=None, languages=["eng-Latn"],
                n_parameters=None, memory_usage_mb=None,
                max_tokens=float(SEQ), embed_dim=self.hidden,
                license=None, open_weights=True, public_training_code=None,
                public_training_data=None, similarity_fn_name="cosine",
                # True when this model takes a task prefix, which is a real
                # property of nomic and not of the BERT four.
                use_instructions=bool(self.prefix), training_datasets=None,
                framework=[], reference=None,
                # required by mteb 2.x ModelMeta; this model is constructed
                # directly, never loaded from the hub, so it is a no-op that
                # returns the live instance.
                loader=lambda **kw: self,
            )
        return self._meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, v):
        self._meta = v

    # ------------------------------------------------------------------ mteb
    # mteb 2.x calls encode(inputs: DataLoader[BatchedInput], *, task_metadata,
    # hf_split, hf_subset, prompt_type, **kwargs), where each batch is a dict
    # with a "text" key (TextInput / CorpusInput / QueryInput all carry it).
    # Anything else -- a plain list of strings -- goes down the simple path
    # above, so the self-test and any direct use keep working.
    def _texts_from(self, inputs):
        if isinstance(inputs, dict):
            return list(inputs["text"])
        if isinstance(inputs, (list, tuple)):
            if inputs and isinstance(inputs[0], dict):
                return [d.get("text", "") for d in inputs]
            return list(inputs)
        texts = []
        for batch in inputs:                      # DataLoader
            if isinstance(batch, dict):
                texts.extend(list(batch["text"]))
            else:
                texts.extend(list(batch))
        return texts

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, batch_size=None, **kwargs):
        texts = self._texts_from(inputs)
        return self._encode_texts(texts)

    def similarity(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        # Embeddings leave the runtime L2-normalised, so cosine is a dot
        # product; normalise defensively anyway so this is correct if that
        # ever changes.
        a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        return a @ b.T

    def similarity_pairwise(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        return (a * b).sum(axis=-1)

    # Older/other call sites
    def encode_queries(self, queries, **kw):
        return self._encode_texts(self._texts_from(queries))

    def encode_corpus(self, corpus, **kw):
        if isinstance(corpus, list) and corpus and isinstance(corpus[0], dict):
            texts = [(c.get("title", "") + " " + c.get("text", "")).strip()
                     for c in corpus]
        else:
            texts = self._texts_from(corpus)
        return self._encode_texts(texts)


def _selftest() -> int:
    """Encode the golden corpus and compare against the CPU model at seq 64."""
    sys.path.insert(0, str(REPO / "reference"))
    from corpus import SENTENCES

    npu = NpuEncoder(verbose=True)
    got = npu.encode(list(SENTENCES))

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(REPO / "models" / "all-MiniLM-L6-v2"),
                             device="cpu")
    st.max_seq_length = SEQ
    ref = st.encode(list(SENTENCES), convert_to_numpy=True,
                    normalize_embeddings=True)

    cos = (got.astype(np.float64) * ref.astype(np.float64)).sum(axis=1)
    print(f"\n  per-sentence 1-cos vs sentence-transformers (seq {SEQ}):")
    for i, c in enumerate(cos):
        print(f"    [{i}] {1.0 - c:.3e}")
    worst = float(1.0 - cos.min())
    print(f"  worst 1-cos {worst:.3e}")
    ok = worst < 2e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
