# SPDX-License-Identifier: Apache-2.0
"""Build a torch model from the ONNX export, not from a HF checkpoint.

WHY THIS EXISTS. The int8/int4 calibration for the ViT and Whisper packers used
to call `from_pretrained(model_dir)`. That needs `model.safetensors` or
`pytorch_model.bin` in the model directory, and the ONNX migration deleted every
one of them: ONNX is now the only copy of the weights this repository has. So
`--dtype i8` and `--dtype i4` died with an unhandled

    OSError: Error no file named pytorch_model.bin, model.safetensors,
    tf_model.h5, model.ckpt.index or flax_model.msgpack found in directory ...

for all seven of those models -- a traceback from deep inside transformers that
never mentions that the weights are present, one directory up, in the ONNX.
The BERT family never hit it because `calibrate_smoothing` runs the numpy
oracle (`reference/encoder.py`) over the ONNX and only asks transformers for a
tokenizer. This module is the same idea for the architectures whose calibration
wants real hooked activations.

IT IS NOT `from_pretrained` WITH EXTRA STEPS. The construction here is
deliberately loud about the three ways a partial load produces plausible
numbers instead of an error:

  * an ONNX tensor that maps to no parameter -- a renamed module, a head the
    export dropped, a prefix that stopped matching;
  * a parameter left at its RANDOM INITIALISATION, because nothing filled it;
  * a tensor whose shape disagrees with the parameter's.

Any of the three means the calibration would measure the activations of a model
that is not this checkpoint, and the smoothing scales would be smoothly wrong:
the int8 container would pack, load, run, and quietly lose accuracy. So all
three refuse, by name, before any forward pass runs.

THE NAME MAPPINGS ARE MEASURED, NOT GUESSED. `verify_onnx_torch_names()`
re-derives the set comparison this module relies on, and the check is SET
EQUALITY against the real `state_dict()`, so a transformers release that
renames a submodule is caught here rather than as a random-weight calibration.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

from onnx_weights import OnnxWeights                              # noqa: E402


class OnnxTorchMismatch(SystemExit):
    """The ONNX export and the torch module tree disagree. Named, never guessed."""


def _mapped_names(sources) -> set[str]:
    """Every ONNX tensor name, prefixed, across all `sources`.

    Deprecated in favour of `_weights` / `_constants`, which split the same set
    in two. Kept because a caller with no module tree in hand -- checking that
    two exports agree, say -- genuinely wants the undivided set.
    """
    out: set[str] = set()
    for reader, prefix in _sources(sources):
        out.update(prefix + t for t in reader._order)
    return out


# HOW AN INITIALIZER IS CLASSIFIED AS A WEIGHT OR A FOLDED CONSTANT.
#
# GraphProto.initializer holds two different things. One is a weight: nothing
# computes it, the graph reads it. The other is a compile-time value the
# exporter folded into an initializer -- Constant and ConstantOfShape outputs,
# the rope cache a Concat builds, an Unsqueeze's axes vector, a scalar scale.
# Both arrive as field 5, so the field number cannot tell them apart, and a
# folded constant has no counterpart in a torch module tree by construction.
#
# The obvious test -- "did a node in this graph produce this name?" -- DOES NOT
# WORK, and not as a subtlety. optimum's exporter folds a Constant node's output
# into an initializer and then REWIRES the consumer to read the initializer, so
# the folded name is never any node's output. Measured on whisper-medium's
# decoder: 602 initializers, 0 of them named by a node output, 30 of them
# folded constants. `OnnxWeights.is_computed` records the question and answers
# it; the answer is "no" for all of them.
#
# So the split is made against the module tree, which is the thing that has to
# be filled, by two rules in order:
#
#   1. the name is a parameter of the tree  -> a weight, always correct;
#   2. the name ends in `.weight` or `.bias` -> a weight, and the strict
#      "extra" check below then demands a parameter for it.
#
# Rule 2 is what keeps this from being a loophole. A real weight whose recovered
# name were not parameter-shaped would fall to "constant" and be skipped -- but
# its parameter would then get no tensor, and the GAP check refuses by name. So
# a weight cannot be silently dropped by being misclassified: the two checks
# bracket it from both sides. A genuinely weight-shaped name with no parameter
# is caught by the other direction, which is the case a wrong `prefix` produces.
_PARAM_SUFFIXES = (".weight", ".bias")


def _split(reader, prefix: str, state) -> tuple[list[str], list[str]]:
    """(weight names, folded-constant names) for one source, both prefixed."""
    weights, consts = [], []
    for tname in reader._order:
        key = prefix + tname
        if key in state or key.endswith(_PARAM_SUFFIXES):
            weights.append(tname)
        else:
            consts.append(tname)
    return weights, consts


def _sources(sources):
    """sources, each resolved to an OnnxWeights. The readers, once."""
    return [(r if isinstance(r, OnnxWeights) else OnnxWeights(r), p)
            for r, p in sources]


def _weights(sources, state) -> set[str]:
    """Every prefixed name in these sources that must reach a parameter."""
    out: set[str] = set()
    for reader, prefix in _sources(sources):
        w, _ = _split(reader, prefix, state)
        out.update(prefix + t for t in w)
    return out


def _constants(sources, state) -> set[str]:
    """The folded compile-time values these sources carry, prefixed.

    Kept beside _weights so the two are read together: one is what a module
    tree must cover, the other is what it must NOT be asked to cover.
    """
    out: set[str] = set()
    for reader, prefix in _sources(sources):
        _, c = _split(reader, prefix, state)
        out.update(prefix + t for t in c)
    return out


def _build_module_tree(model_dir: Path, config_cls: str, model_cls: str):
    """(model, state_dict) from config.json alone -- no weights, all random."""
    import transformers

    cfg_attr = getattr(transformers, config_cls, None)
    mdl_attr = getattr(transformers, model_cls, None)
    if cfg_attr is None or mdl_attr is None:
        raise OnnxTorchMismatch(
            f"transformers has no {config_cls}/{model_cls}, which the ONNX "
            f"export for {model_dir.name} is named after. The calibration "
            f"builds the module tree from those classes; without them there is "
            f"no tree to hook, and guessing a substitute would hook the wrong "
            f"modules.")

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise OnnxTorchMismatch(
            f"{model_dir}/config.json is missing; the calibration builds the "
            f"torch module tree from it, and nothing else records how deep this "
            f"checkpoint is.")
    model = mdl_attr(cfg_attr.from_json_file(str(cfg_path)))
    return model, model.state_dict()


def _mem_available() -> int | None:
    """Bytes of RAM this process could still get, or None if unmeasurable.

    MemAvailable, not MemFree: the kernel's own estimate of what is reclaimable
    including page cache, which is the number that predicts whether an
    allocation succeeds. `ru_maxrss` would be peak usage, which says nothing
    about what is left.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# How much free RAM a calibration wants left over once its module tree exists.
# The tree is already allocated by the time this runs; what is still to come is
# the ONNX weight read (one tensor at a time, transient) and one forward pass
# with no_grad, whose activations are proportional to the corpus and not to the
# model. A fifth of the tree is a generous allowance for both on a model this
# size, and the number only has to be right enough to turn a silent SIGKILL
# into a sentence that names the model.
_HEADROOM = 0.2


def _check_memory(model, label: str) -> None:
    """Refuse by name if the forward pass will not fit, instead of being killed.

    The OOM killer sends SIGKILL, which the interpreter cannot catch: the
    process stops between two bytecodes and prints nothing at all -- not a
    traceback, not a partial line. That is why whisper-large-v3's int8
    calibration first appeared here as an empty message: a fact with no cause
    attached, on a 13 GB machine, for a model whose module tree is 5.3 GiB.

    Called AFTER the tree is built (its size is the thing being measured) and
    BEFORE the weights are read into it (the expensive part), so that the
    refusal arrives while there is still memory to refuse with. It is a check,
    not a reservation, and it never refuses a tree that plainly fits.
    """
    avail = _mem_available()
    if avail is None:
        return
    need = sum(p.numel() * p.element_size() for p in model.parameters())
    if need * _HEADROOM > avail:
        raise OnnxTorchMismatch(
            f"{label}: the module tree is {need / 2**30:.1f} GiB and this "
            f"machine has {avail / 2**30:.1f} GiB free, which is not enough "
            f"left to read the ONNX weights into it and run the forward pass "
            f"the calibration needs. The OOM killer would stop this process "
            f"with no message at all, which is how it used to present. Free "
            f"memory or use a larger machine; the weights are filled in place "
            f"into a tree that already exists, so there is no smaller path "
            f"through this.")


def build(model_dir: str | Path, config_cls: str, model_cls: str,
          sources, *, allow_missing=(), what: str = ""):
    """A torch model whose every parameter came from the ONNX export.

    `sources` is an iterable of `(reader, prefix)` pairs: `reader` is an
    `OnnxWeights` (or a path to one) and `prefix` is prepended to each tensor
    name it hands back. The prefix is what the EXPORT drops, not what the model
    is called -- whisper's `encoder_model.onnx` names a tensor `conv1.weight`
    where the module tree has `model.encoder.conv1.weight`, while
    `decoder_model.onnx` already carries `model.decoder.` in its own names and
    must NOT be prefixed again. Prefixing both is a real mistake this comment
    exists for: it yields `model.decoder.model.decoder.embed_positions.weight`,
    which matches nothing and, before the unmapped-tensor refusal below, would
    have left the entire decoder randomly initialised.

    `allow_missing` names parameters this export legitimately does not carry --
    whisper's decoder-only export has no `proj_out`, because that projection is
    the tied `embed_tokens` and lives in the encoder graph. Each is still named
    in the printout, because "we skipped N" is the sentence that makes an
    incomplete load auditable rather than silent.

    ONLY THE WEIGHTS ARE CHECKED, which is a distinction the export forces.
    GraphProto.initializer holds folded compile-time values as well as weights
    -- Constant and ConstantOfShape outputs, the rope cache a Concat builds, an
    Unsqueeze's axes -- and those have no counterpart in a module tree by
    construction. whisper-tiny's export folds none and matches its tree exactly;
    whisper-medium folds 30 and whisper-large-v3-turbo 603, so a check over
    every initializer reads as "this model is broken" for reasons that have
    nothing to do with the model. `_split` separates them; see the note there
    for why provenance does not. The count of what was set aside is printed,
    because a calibration that quietly drops 603 tensors deserves to say so.
    """
    import torch

    mp = Path(model_dir)
    model, state = _build_module_tree(mp, config_cls, model_cls)
    _check_memory(model, what or mp.name)
    sources = _sources(sources)
    mapped = _weights(sources, state)

    unknown = sorted(mapped - set(state))
    if unknown:
        raise OnnxTorchMismatch(
            f"{what or mp.name}: {len(unknown)} ONNX weight(s) map to no "
            f"parameter, first: {unknown[:4]}. Either the export's naming moved "
            f"or the `prefix` passed here is wrong. Loading the rest would leave "
            f"those parameters at their random initialisation and the "
            f"calibration would measure the wrong model, so this refuses.")

    missing = sorted(set(state) - mapped - set(allow_missing))
    if missing:
        raise OnnxTorchMismatch(
            f"{what or mp.name}: {len(missing)} parameter(s) got no ONNX "
            f"tensor, first: {missing[:4]}. They would stay randomly "
            f"initialised. If this export legitimately omits them, name them in "
            f"allow_missing -- do not widen this check.")

    # FILLED IN PLACE, NOT INTO A SECOND COPY OF THE MODEL.
    #
    # `state_dict()` hands back the live parameter tensors, so writing into them
    # with copy_ puts the weights where they belong and costs one tensor of
    # transient space. Building a `filled` dict of torch tensors first and
    # handing it to load_state_dict is the obvious spelling and costs the whole
    # model twice: the module tree, then a full second copy in `filled`, then a
    # third transient inside the copy. For the four models in this repository
    # that is invisible. For whisper-large-v3 -- 1.43 B parameters, a 5.3 GiB
    # fp32 module tree -- it is the difference between calibrating and being
    # killed by the OOM reaper, which is how this first presented: an empty
    # message and no traceback, because SIGKILL leaves nothing to print.
    #
    # So the weights go straight into the parameters they belong to.
    n_filled = 0
    for reader, prefix in sources:
        weights, _consts = _split(reader, prefix, state)
        for tname in weights:
            key = prefix + tname
            dst = state[key]
            arr = reader.raw(tname)
            want, got = tuple(dst.shape), tuple(arr.shape)
            if got != want:
                raise OnnxTorchMismatch(
                    f"{what or mp.name}: ONNX tensor {key!r} is {got} but the "
                    f"module tree's parameter is {want}. The export and this "
                    f"transformers version disagree about the layer's shape; "
                    f"refusing rather than loading something that would "
                    f"broadcast.")
            # `torch.from_numpy` on a read-only mmap view warns once per
            # process and then suppresses every later warning, which is a bad
            # trade for a calibration that should print its corpus and its site
            # count and nothing else. `copy_` reads it as C-contiguous when it
            # has to, so no writable copy is needed first -- and copy_ also
            # does the dtype conversion, which is why the target's dtype is not
            # asked for separately.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dst.copy_(torch.from_numpy(arr))
            n_filled += 1

    skipped = sorted(set(allow_missing) & set(state))
    label = what or mp.name
    n_const = len(_constants(sources, state))
    tail = (f"; {n_const} folded graph constant(s) set aside"
            if n_const else "")
    if skipped:
        print(f"  {label}: {n_filled} weight tensors from ONNX; "
              f"{len(skipped)} named absent from this export "
              f"({', '.join(skipped[:4])}){tail}")
    else:
        print(f"  {label}: {n_filled} weight tensors from ONNX -- every "
              f"parameter of the module tree{tail}")
    return model.eval()


def verify_onnx_torch_names(model_dir: str | Path, config_cls: str,
                            model_cls: str, sources, *,
                            allow_missing=()) -> tuple[int, int]:
    """(matched, total) after checking set equality in BOTH directions.

    Exists so the mappings above are checkable on their own, without building a
    model or running a forward pass. The failure it catches -- a transformers
    release renaming a submodule -- is silent otherwise, because a renamed
    module simply stops being filled.

    `matched` counts weights, not initializers, for the reason spelled out at
    `_split`. The two directions are what make the split safe to rely on: a
    weight misclassified as a constant shows up as a gap, and a weight-shaped
    name with no parameter shows up as an extra.
    """
    mp = Path(model_dir)
    _, state = _build_module_tree(mp, config_cls, model_cls)
    sources = _sources(sources)
    mapped = _weights(sources, state)
    extra = sorted(mapped - set(state))
    if extra:
        raise OnnxTorchMismatch(
            f"{mp.name}: ONNX has {len(extra)} weight(s) the module tree does "
            f"not, first: {extra[:4]}. A wrong `prefix` is the usual cause: each "
            f"source is prefixed independently, and the encoder and decoder "
            f"exports do not want the same treatment.")
    gap = sorted(set(state) - mapped - set(allow_missing))
    if gap:
        raise OnnxTorchMismatch(
            f"{mp.name}: module tree has {len(gap)} parameter(s) the ONNX does "
            f"not, first: {gap[:4]}. Name them in allow_missing if this export "
            f"legitimately omits them.")
    return len(mapped), len(state)
