//===- head.hpp ----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the ViT classifier head, on the host. arch=5.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//
//
// ITS OWN TRANSLATION UNIT, AND WHY
// -------------------------------
// This is 12 lines of arithmetic, and it is a file of its own for the same
// reason vit/image.cpp is: it is the part of the arch=5 path a host-only box
// can actually falsify. As a member of VitEncoder it was unreachable without a
// container, a design set and a device, and the one thing it can get wrong is
// SILENT -- see below. tools/verify/verify_vit_model.py builds
// runtime/tests/test_vit_model.cpp against vit/head.cpp alone, with no NPU, no
// XRT and no npu_ops, because pulling in vit/encoder.cpp would link the whole
// AIE stack for a 768 x 1000 matvec.
//
// THE STRIDE IS THE WHOLE FILE
// ----------------------------
// `classifier.weight` is DECLARED [d_model, num_labels] and stored row-major,
// so element (t, label) is at head[t * num_labels + label]: the label is the
// fast axis, and one label's 768 weights are a stride-1000 walk through a 3 MB
// table.
//
// The checkpoint holds it the other way round. torch's nn.Linear stores a
// [num_labels, d_model] weight, the packer transposes it on the way in so the
// declared shape is honest, and everything after that -- the byte count, the
// layout hash, the tile count, every shape check in read_geometry -- agrees
// either way. Reading it as [num_labels, d_model] therefore computes a finite,
// plausible, entirely WRONG set of 1000 logits, and the argmax of those is
// another label, with the same confidence as a right one. There is no tolerance
// anywhere in the stack that would catch it.
//
// Hence `head_wrong_stride` in the probe: the transpose is computed on purpose
// so the gate can require the two to DIFFER. A gate that only compared against
// a reference could not tell a correct head from a probe printing zeros twice.
//
// THE LABEL RANGE
// ---------------
// [j0, j1) rather than the whole vector, because VitEncoder::classify spreads
// the labels across the thread pool and a gate that only ever exercised the
// single-threaded path would not know the split is sound. The blocks are
// contiguous rather than interleaved for a cache reason stated at the call
// site; the arithmetic does not care, which is the point:
//
//   one label's sum always runs t = 0 .. d_model-1, in that order, into one
//   double, so label j's bytes do not depend on which worker owns it or how
//   many workers there are.
//
// The accumulator is a double and the products are cast to double, for the
// same reason the NPU's C panels are fp32: 768 float32 additions of mixed-sign
// terms accumulate more error than the output is worth at an argmax boundary.
//
// The refusal below is not defensive padding. `classifier.weight` is the ONE
// operand in the whole arch=5 stack that never reaches the device, so there is
// no other place that would notice it was not read -- and "label 0's weight
// applied to all 1000 labels" is another silent wrong answer.

#pragma once

#include <cstdint>

namespace npue::vit {

// logits[j] = bias[j] + sum_t cls_row[t] * head[t * num_labels + j], for
// j in [j0, j1). `bias` may be null, which is zero.
//
// ALL THREE ARE INDEXED BY THE ABSOLUTE LABEL j, NOT BY j - j0. `logits` is the
// base of the WHOLE array even when the block is one label wide. This is not
// restating the obvious: the first version of the probe passed `logits + j`
// with a one-label block, read past the end of the vector and died in malloc.
// bias[j] and head[t * num_labels + j] cannot be block-relative without an
// offset on each, so the block is a RANGE on one shared index space rather than
// three pointers that each mean something different.
//
// REFUSES, by name: a null `head`, `cls_row` or `logits`; a non-positive
// d_model or num_labels; and a [j0, j1) that is not inside [0, num_labels).
// An out-of-range block is not clamped -- clamping would silently write fewer
// logits than the caller asked for and leave the rest at whatever they were.
void head_matvec(const float *head, const float *bias, const float *cls_row,
                 int64_t d_model, int64_t num_labels, int64_t j0, int64_t j1,
                 float *logits);

}  // namespace npue::vit
