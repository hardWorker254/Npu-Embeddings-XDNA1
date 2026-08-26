// Reproduction of tasks/0045's narrowing probe (fp32 -> bf16), 64 elements,
// -O2, llvm-objdump. Confirms this session's compile setup reproduces the
// 34-vs-0 vmul.f/vadd.f result before it is used to judge gelu_poly.cc.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

extern "C" {

void narrow_mulby1_64(float *restrict in, bfloat16 *restrict out) {
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);
  auto it_in = aie::begin_restrict_vector<16>(in);
  auto it_out = aie::begin_restrict_vector<16>(out);
  for (int i = 0; i < 64; i += 16) {
    *it_out++ = aie::mul(*it_in++, vone_f).to_vector<bfloat16>();
  }
}

void narrow_accum_64(float *restrict in, bfloat16 *restrict out) {
  auto it_in = aie::begin_restrict_vector<16>(in);
  auto it_out = aie::begin_restrict_vector<16>(out);
  for (int i = 0; i < 64; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(*it_in++);
    *it_out++ = a.to_vector<bfloat16>();
  }
}

}  // extern "C"
