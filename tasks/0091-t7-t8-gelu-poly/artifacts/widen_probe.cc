// T7 probe: isolate the cost of bf16 -> fp32 WIDENING alone, same shape as
// tasks/0045's narrowing probe (64 elements, -O2, llvm-objdump). Two forms:
//   mulby1:  aie::mul(v, vone_bf).to_vector<float>()      (gelu_poly.cc today)
//   accum:   accum<accfloat,16> a; a.from_vector(v); a.to_vector<float>()
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

extern "C" {

void widen_mulby1_64(bfloat16 *restrict in, float *restrict out) {
  const aie::vector<bfloat16, 16> vone_bf = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  auto it_in = aie::begin_restrict_vector<16>(in);
  auto it_out = aie::begin_restrict_vector<16>(out);
  for (int i = 0; i < 64; i += 16) {
    *it_out++ = aie::mul(*it_in++, vone_bf).to_vector<float>();
  }
}

void widen_accum_64(bfloat16 *restrict in, float *restrict out) {
  auto it_in = aie::begin_restrict_vector<16>(in);
  auto it_out = aie::begin_restrict_vector<16>(out);
  for (int i = 0; i < 64; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(*it_in++);
    *it_out++ = a.to_vector<float>();
  }
}

}  // extern "C"
