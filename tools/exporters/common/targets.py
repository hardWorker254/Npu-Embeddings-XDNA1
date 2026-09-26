"""tools/npu_targets.json: the exporters' source of truth for model geometry.

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
    Read and validate tools/npu_targets.json.

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
        _require_int(spec, "hidden", ctx)
        _require_int(spec, "intermediate", ctx)
        if not isinstance(spec.get("gated_ffn"), bool):
            raise SystemExit(f"{ctx}: 'gated_ffn' must be a boolean")
        if kind == "stt" and spec["gated_ffn"]:
            raise SystemExit(
                f"{ctx}: gated_ffn is true. Whisper's FFN is a plain GELU MLP; "
                f"a gated entry here would export a 2*intermediate ffn_up the "
                f"model does not have.")
        _require_int(spec, "qkv_n", ctx, allow_none=True)
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
