"""tools/data/npu_targets.json: the exporters' source of truth for model geometry.

One file, read by both exporters and mirrored by the runtime catalogue in
src/common/hub.cpp. Schema validation is deliberately strict: a typo in a
geometry field must fail here, not silently export a design for the wrong shape.
"""

import json
from pathlib import Path

from .consts import (
    KNOWN_DATAPATHS,
    KNOWN_DEFAULT_KEYS,
    KNOWN_KINDS,
    KNOWN_MODEL_KEYS,
    STT_MODEL_KEYS,
    TARGETS_SCHEMA,
)
from .validate import _require_int

def load_targets(path: str | Path) -> dict:
    """
    Read and validate tools/data/npu_targets.json.

    Validation is structural only: schema version, known keys and value types.
    Parity with the C++ catalog in runtime/src/common/hub.cpp is a separate
    check (tools/verify_targets.py, S7).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"targets file not found: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{p}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"{p}: top level must be an object")

    schema = data.get("schema")
    if schema != TARGETS_SCHEMA:
        raise SystemExit(
            f"{p}: unsupported schema {schema!r}; expected {TARGETS_SCHEMA}"
        )

    defaults = data.get("defaults")
    arches = data.get("arches")
    models = data.get("models")
    kinds = data.get("kinds")

    if not isinstance(defaults, dict):
        raise SystemExit(f"{p}: missing or invalid 'defaults' object")
    if not isinstance(arches, dict) or not arches:
        raise SystemExit(f"{p}: missing or empty 'arches' object")
    if not isinstance(models, dict) or not models:
        raise SystemExit(f"{p}: missing or empty 'models' object")
    if kinds is not None and not isinstance(kinds, dict):
        raise SystemExit(f"{p}: 'kinds' must be an object")
    for kname, kspec in (kinds or {}).items():
        kctx = f"{p}: kinds[{kname!r}]"
        if not isinstance(kspec, dict):
            raise SystemExit(f"{kctx}: must be an object")
        unknown = set(kspec) - {"exporter", "streams", "note"}
        if unknown:
            raise SystemExit(f"{kctx}: unknown keys: {sorted(unknown)}")
        streams = kspec.get("streams")
        if not isinstance(streams, list) or any(
                not isinstance(s, str) or not s for s in streams):
            raise SystemExit(f"{kctx}: 'streams' must be a list of names")
        if len(set(streams)) != len(streams):
            raise SystemExit(f"{kctx}: 'streams' has a duplicate name")
        exporter = kspec.get("exporter")
        if exporter is not None and not isinstance(exporter, str):
            raise SystemExit(f"{kctx}: 'exporter' must be a module name or null")

    unknown_defaults = set(defaults) - KNOWN_DEFAULT_KEYS
    if unknown_defaults:
        raise SystemExit(
            f"{p}: unknown defaults keys: {sorted(unknown_defaults)}"
        )
    for key in ("seq", "tile_m", "tile_k", "tile_n", "identity_threshold"):
        _require_int(defaults, key, f"{p}: defaults")
    if "c_bf16" in defaults and not isinstance(defaults["c_bf16"], bool):
        raise SystemExit(f"{p}: defaults.c_bf16 must be a boolean")

    for arch, spec in arches.items():
        ctx = f"{p}: arches[{arch!r}]"
        if not isinstance(spec, dict):
            raise SystemExit(f"{ctx}: must be an object")
        if not str(arch).isdigit():
            raise SystemExit(f"{ctx}: architecture key must be numeric")
        _require_int(spec, "cols", ctx)
        _require_int(spec, "batch", ctx)
        if "batches" in spec:
            batches = spec["batches"]
            if (
                not isinstance(batches, list)
                or not batches
                or any(
                    not isinstance(b, int) or isinstance(b, bool) or b <= 0
                    for b in batches
                )
            ):
                raise SystemExit(
                    f"{ctx}: 'batches' must be a non-empty list of "
                    f"positive integers"
                )
        _require_int(spec, "max_batch", ctx, allow_none=True)

    for name, spec in models.items():
        ctx = f"{p}: models[{name!r}]"
        if not isinstance(spec, dict):
            raise SystemExit(f"{ctx}: must be an object")
        unknown = set(spec) - KNOWN_MODEL_KEYS
        if unknown:
            raise SystemExit(f"{ctx}: unknown keys: {sorted(unknown)}")
        kind = spec.get("kind", "gemm_rtp")
        if kind not in KNOWN_KINDS:
            raise SystemExit(
                f"{ctx}: 'kind' must be one of {sorted(KNOWN_KINDS)}, got "
                f"{kind!r}. A kind with no exporter is refused by name, not "
                f"silently treated as an embedder.")
        if kind == "stt":
            missing = STT_MODEL_KEYS - set(spec)
            if missing:
                raise SystemExit(
                    f"{ctx}: kind 'stt' needs {sorted(missing)}. An STT entry "
                    f"without its mel bins or layer counts describes no "
                    f"geometry at all.")
            for key in ("heads", "head_dim", "enc_layers", "dec_layers",
                        "mel_bins", "frames", "max_target"):
                _require_int(spec, key, ctx)
            if spec["heads"] * spec["head_dim"] != spec["hidden"]:
                raise SystemExit(
                    f"{ctx}: heads*head_dim = {spec['heads'] * spec['head_dim']}"
                    f" is not hidden = {spec['hidden']}")
        # A POSE target has no hidden width and no FFN. Its streams are twenty-one
        # convolutions, so there is nothing for `hidden` or `intermediate` to
        # describe, and requiring them here would mean writing a number that
        # nothing reads -- the pose export path builds no eltwise design and no
        # LayerNorm, so the only consumer of `hidden` (export_eltwise, reached
        # only when --npu-ops asks for an op) never runs for it. A required
        # field whose value is fiction is worse than an absent one, so this is
        # the one kind that does not carry them, and the check below is what
        # says so rather than letting a missing key read as 0.
        if kind != "pose":
            _require_int(spec, "hidden", ctx)
            _require_int(spec, "intermediate", ctx)
        if not isinstance(spec.get("gated_ffn", False), bool):
            raise SystemExit(f"{ctx}: 'gated_ffn' must be a boolean")
        if kind == "stt" and spec["gated_ffn"]:
            raise SystemExit(
                f"{ctx}: gated_ffn is true. Whisper's FFN is a plain GELU MLP; "
                f"a gated entry here would export a 2*intermediate ffn_up the "
                f"model does not have.")
        if kind != "pose":
            _require_int(spec, "qkv_n", ctx, allow_none=True)
        if kind == "cls":
            # The attention scale is compiled into Q's weight, and it is
            # head_dim**-0.5 -- so a cls entry that omits the head geometry, or
            # whose heads do not tile the hidden width, describes a scale nobody
            # computed. Refused here rather than defaulted, for the same reason
            # the stt branch above refuses a missing mel_bins.
            missing = {"heads", "head_dim", "enc_layers"} - set(spec)
            if missing:
                raise SystemExit(
                    f"{ctx}: kind 'cls' needs {sorted(missing)}. An image "
                    f"classifier's head geometry is what fixes the folded "
                    f"attention scale, so it is not optional.")
            for key in ("heads", "head_dim", "enc_layers"):
                _require_int(spec, key, ctx)
            if spec["heads"] * spec["head_dim"] != spec["hidden"]:
                raise SystemExit(
                    f"{ctx}: heads*head_dim = "
                    f"{spec['heads'] * spec['head_dim']} is not hidden = "
                    f"{spec['hidden']}. The attention scale is folded from "
                    f"head_dim, so this would silently fold the wrong number.")
            if spec["gated_ffn"]:
                raise SystemExit(
                    f"{ctx}: gated_ffn is true. A ViT's FFN is a plain GELU MLP; "
                    f"a gated entry here would export a 2*intermediate ffn_up "
                    f"the model does not have.")
            # The position count is fixed by the image size and the patch, and
            # `seq` is the per-dispatch ROW TILE the runtime pads up to -- not a
            # text sequence length. It still has to be a multiple of 8 (the
            # runtime refuses anything else) and at least as large as the true
            # position count, or the last patches are cut off rather than
            # padded. n_patches is therefore needed to check it.
            for key in ("image_size", "patch_size"):
                if key not in spec:
                    raise SystemExit(
                        f"{ctx}: kind 'cls' needs {key!r}. The position count is "
                        f"(image_size/patch_size)^2 + 1 and the cls entry's "
                        f"seq is the tile it is padded into; without them there "
                        f"is nothing to check the tile against.")
                _require_int(spec, key, ctx)
            n_pos = (spec["image_size"] // spec["patch_size"]) ** 2 + 1
            seq = (spec.get("overrides") or {}).get("seq")
            if seq is not None and seq < n_pos:
                raise SystemExit(
                    f"{ctx}: overrides['seq'] is {seq} but this model has {n_pos} "
                    f"positions at patch {spec['patch_size']} over "
                    f"{spec['image_size']}px. The runtime pads rows up to seq, "
                    f"so a smaller tile would drop positions rather than pad "
                    f"them.")
            # patch_size must tile the image, or the im2col is a strided window
            # with a remainder and the exact-GEMM rewrite is not exact.
            if spec["image_size"] % spec["patch_size"]:
                raise SystemExit(
                    f"{ctx}: image_size {spec['image_size']} is not a multiple "
                    f"of patch_size {spec['patch_size']}. The patch-embedding "
                    f"conv is rewritten as an exact GEMM on the grounds that "
                    f"stride equals kernel with no remainder.")
        datapath = spec.get("datapath")
        if datapath is not None and datapath not in KNOWN_DATAPATHS:
            raise SystemExit(
                f"{ctx}: 'datapath' must be one of {sorted(KNOWN_DATAPATHS)}"
            )
        overrides = spec.get("overrides")
        if overrides is not None:
            if not isinstance(overrides, dict):
                raise SystemExit(f"{ctx}: 'overrides' must be an object")
            unknown = set(overrides) - KNOWN_DEFAULT_KEYS
            if unknown:
                raise SystemExit(
                    f"{ctx}: unknown overrides keys: {sorted(unknown)}"
                )
            # `batches` is a LIST, and every other override is a scalar. It is
            # checked here rather than left to resolve_args so a typo in a tier
            # list is a message about the file, not a shape error twenty
            # minutes into an AIE build.
            if "batches" in overrides:
                b = overrides["batches"]
                if (
                    not isinstance(b, list)
                    or not b
                    or any(
                        not isinstance(x, int) or isinstance(x, bool) or x <= 0
                        for x in b
                    )
                ):
                    raise SystemExit(
                        f"{ctx}: overrides['batches'] must be a non-empty list "
                        f"of positive integers, got {b!r}"
                    )
                if "batch" in overrides and overrides["batch"] != max(b):
                    raise SystemExit(
                        f"{ctx}: overrides['batch'] ({overrides['batch']}) must "
                        f"be the largest tier in overrides['batches'] ({b})"
                    )
            for key, value in overrides.items():
                if key == "batches":
                    continue
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise SystemExit(
                        f"{ctx}: overrides[{key!r}] must be a non-negative "
                        f"integer"
                    )

    return data

def print_targets(targets: dict) -> None:
    print(f"targets file (schema {targets['schema']})")
    print("arches:")
    for arch, spec in sorted(targets["arches"].items()):
        print(
            f"  {arch}: device={spec.get('device')} cols={spec.get('cols')} "
            f"batch={spec.get('batch')} batches={spec.get('batches')} "
            f"max_batch={spec.get('max_batch')}"
        )
    print("models:")
    for name in sorted(targets["models"]):
        spec = targets["models"][name]
        qkv = spec.get("qkv_n")
        print(
            f"  {name}: hidden={spec.get('hidden')} "
            f"intermediate={spec.get('intermediate')} "
            f"gated_ffn={spec.get('gated_ffn')} "
            f"qkv_n={'3*hidden' if qkv is None else qkv} "
            f"datapath={spec.get('datapath')}"
        )
