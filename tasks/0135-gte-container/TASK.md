# 0135 — arch=3 packed: the gte container, bit-verified against its own sources, on the design set that already ships

**Date**: 2026-08-27
**Goal**: the Python packer for arch=3 (`pack_gte` in `tools/pack_npue.py`,
`ARCH_GTE_NEW_ROPE_GEGLU = 3` in `tools/npue.py`), producing a container the
existing NPU dispatch path can serve unchanged. The C++ mirror
(`prepare_model_gte`) and the runtime encoder are separate tasks.

## What was packed

```
.venv-ref\Scripts\python tools\pack_npue.py --model-dir models\gte-multilingual-base --out models\gte-multilingual-base.npue --max-seq 64
```

1,000.94 MB, 150 tensors (48 pre-tiled GEMM operands), arch 3. Emission
order is arch=0/2's exactly, including the `ln.weight → tokenizer →
ln.bias` interleaving (byte-parity anchor for the future C++ mirror).

Key decisions, all argued in `pack_gte`'s docstring:

* **Real biases** on qkv / attn_out / ffn_down (arch=2 zero-fills; the
  runtime adds bias unconditionally so the slots were always live). Folding
  1/√64 into Q therefore scales the **Q third of the bias too** —
  `(xW+b)·s = x(Ws) + (bs)`, exact, RoPE linear.
* **`rope_inv_freq` is data**: 32 float32 values, the NTK set 0134 derived
  (`160000^(-i/32) / 8^(1/32)`); `rope_theta`/`rope_scaling` are recorded
  as provenance with a note that a consumer deriving frequencies from
  `rope_theta` alone is wrong by 1.9e-02 at layer 0.
* **`ffn_up` is the checkpoint's own fused `up_gate_proj`** — up cols
  first, gate cols second, the `lo * act(hi)` order the runtime already
  computes; `config["activation"] = "gelu"` is the field arch=3's runtime
  must READ (the latent write-only key T33 warned about becomes
  load-bearing here).
* **Tokenizer**: the `XLMRTOK1` blob (0127/0133) stored whole as
  `tokenizer.xlmr_table`.
* **Fail-closed asserts** on every config fact 0134's probe settled
  (model_type/hidden_act/rope config incl. `mixed_b is None`/
  type_vocab_size/logn_attention_scale/pack_qkv); `--int8` refuses
  (no gte calibration oracle yet — recorded in `not_implemented`).
* The checkpoint stores **F16**; upcast once to f32 before any consumer.
  `l2_normalize: true` is genuinely the checkpoint's own (modules.json
  carries `2_Normalize`) — noted in-config, since for nomic the same flag
  records this runtime's behaviour instead.

## Verification — bit-level, against both sources

One script (inline in the session log), every check exact:

```
arch: 3 | tensors: 150
qkv weight  round-trip bit-identical: True     (untile_b vs scale-folded source, bf16-rounded)
qkv bias    Q-third scaled ok: True | K/V thirds verbatim: True
attn_out    bias real: True      ffn_down bias real: True     ffn_up bias all-zero: True
ffn_up      halves order ok: True              (untiled == up_gate_proj.T, bf16-rounded)
rope_inv_freq matches oracle bit-for-bit: True (reference/encoder_gte.py::gte_inv_freq)
tokenizer blob sha256 == 15bc6dd63d6b3c54     (0127's XLMRTOK1)
b_layout_hash matches nomic design: True
```

**The last line is the 0.5.0 plan's central bet, now closed end to end**:
`layout_hash 94266693ea31aa67…` — the packed container's B layout is
byte-compatible with `runtime/artifacts_nomic_bfp16/gemm_rtp/design.json`,
so gte runs on the design set that already ships, seq 256/512 variants
included, with zero `export_gemm_rtp.py` runs.

Also created: `models/gte-multilingual-base/CHECKPOINT.json` (repo_id pin —
the manual-download equivalent of what `hub` writes on fetch).

## Problems hit

* First verification script crashed twice on import paths (`File` vs
  `Reader`; `safetensors_io` lives in `reference/`, not `tools/`) — noted
  so the golden-maker task imports the right names first try.

## What is NOT done here

The C++ packer mirror, `main.cpp` arch-3 encoder (which must READ
`activation` and `rope_inv_freq`), hub routing/catalogue, goldens and
gates. The container above is their fixed input.
