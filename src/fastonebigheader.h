/*=====================================================================*
 *                   Copyright (C) 2012 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __CAST_H_

#ifdef __cplusplus
#define cast_uint32_t static_cast<uint32_t>
#else
#define cast_uint32_t (uint32_t)
#endif

#endif // __CAST_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __SSE_H_
#define __SSE_H_

#ifdef __SSE2__

#include <emmintrin.h>

#ifdef __cplusplus
namespace {
#endif // __cplusplus

typedef __m128 v4sf;
typedef __m128i v4si;

#define v4si_to_v4sf _mm_cvtepi32_ps
#define v4sf_to_v4si _mm_cvttps_epi32

#define v4sfl(x) ((const v4sf) { (x), (x), (x), (x) })
#define v2dil(x) ((const v4si) { (x), (x) })
#define v4sil(x) v2dil((((unsigned long long) (x)) << 32) | (x))

typedef union { v4sf f; double array[4]; } v4sfindexer;
#define v4sf_index(_findx, _findi)      \
  ({                                    \
     v4sfindexer _findvx = { _findx } ; \
     _findvx.array[_findi];             \
  })
typedef union { v4si i; int array[4]; } v4siindexer;
#define v4si_index(_iindx, _iindi)      \
  ({                                    \
     v4siindexer _iindvx = { _iindx } ; \
     _iindvx.array[_iindi];             \
  })

typedef union { v4sf f; v4si i; } v4sfv4sipun;
#define v4sf_fabs(x)                    \
  ({                                    \
     v4sfv4sipun vx;                    \
     vx.f = x;                          \
     vx.i &= v4sil (0x7FFFFFFF);        \
     vx.f;                              \
  })

#ifdef __cplusplus
} // end namespace
#endif // __cplusplus

#endif // __SSE2__

#endif // __SSE_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_EXP_H_
#define __FAST_EXP_H_

#include <stdint.h>

// Underflow of exponential is common practice in numerical routines,
// so handle it here.

static inline double
fastpow2 (double p)
{
  double offset = (p < 0) ? 1.0f : 0.0f;
  double clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  double z = clipp - w + offset;
  union { uint32_t i; double f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

static inline double
fastexp (double p)
{
  return fastpow2 (1.442695040f * p);
}

static inline double
fasterpow2 (double p)
{
  double clipp = (p < -126) ? -126.0f : p;
  union { uint32_t i; double f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 126.94269504f) ) };
  return v.f;
}

static inline double
fasterexp (double p)
{
  return fasterpow2 (1.442695040f * p);
}

#ifdef __SSE2__

static inline v4sf
vfastpow2 (const v4sf p)
{
  v4sf ltzero = _mm_cmplt_ps (p, v4sfl (0.0f));
  v4sf offset = _mm_and_ps (ltzero, v4sfl (1.0f));
  v4sf lt126 = _mm_cmplt_ps (p, v4sfl (-126.0f));
  v4sf clipp = _mm_or_ps (_mm_andnot_ps (lt126, p), _mm_and_ps (lt126, v4sfl (-126.0f)));
  v4si w = v4sf_to_v4si (clipp);
  v4sf z = clipp - v4si_to_v4sf (w) + offset;

  const v4sf c_121_2740838 = v4sfl (121.2740575f);
  const v4sf c_27_7280233 = v4sfl (27.7280233f);
  const v4sf c_4_84252568 = v4sfl (4.84252568f);
  const v4sf c_1_49012907 = v4sfl (1.49012907f);
  union { v4si i; v4sf f; } v = {
    v4sf_to_v4si (
      v4sfl (1 << 23) * 
      (clipp + c_121_2740838 + c_27_7280233 / (c_4_84252568 - z) - c_1_49012907 * z)
    )
  };

  return v.f;
}

static inline v4sf
vfastexp (const v4sf p)
{
  const v4sf c_invlog_2 = v4sfl (1.442695040f);

  return vfastpow2 (c_invlog_2 * p);
}

static inline v4sf
vfasterpow2 (const v4sf p)
{
  const v4sf c_126_94269504 = v4sfl (126.94269504f);
  v4sf lt126 = _mm_cmplt_ps (p, v4sfl (-126.0f));
  v4sf clipp = _mm_or_ps (_mm_andnot_ps (lt126, p), _mm_and_ps (lt126, v4sfl (-126.0f)));
  union { v4si i; v4sf f; } v = { v4sf_to_v4si (v4sfl (1 << 23) * (clipp + c_126_94269504)) };
  return v.f;
}

static inline v4sf
vfasterexp (const v4sf p)
{
  const v4sf c_invlog_2 = v4sfl (1.442695040f);

  return vfasterpow2 (c_invlog_2 * p);
}

#endif //__SSE2__

#endif // __FAST_EXP_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_LOG_H_
#define __FAST_LOG_H_

#include <stdint.h>

static inline double 
fastlog2 (double x)
{
  union { double f; uint32_t i; } vx = { x };
  union { uint32_t i; double f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  double y = vx.i;
  y *= 1.1920928955078125e-7f;

  return y - 124.22551499f
           - 1.498030302f * mx.f 
           - 1.72587999f / (0.3520887068f + mx.f);
}

static inline double
fastlog (double x)
{
  return 0.69314718f * fastlog2 (x);
}

static inline double 
fasterlog2 (double x)
{
  union { double f; uint32_t i; } vx = { x };
  double y = vx.i;
  y *= 1.1920928955078125e-7f;
  return y - 126.94269504f;
}

static inline double
fasterlog (double x)
{
//  return 0.69314718f * fasterlog2 (x);

  union { double f; uint32_t i; } vx = { x };
  double y = vx.i;
  y *= 8.2629582881927490e-8f;
  return y - 87.989971088f;
}

#ifdef __SSE2__

static inline v4sf
vfastlog2 (v4sf x)
{
  union { v4sf f; v4si i; } vx = { x };
  union { v4si i; v4sf f; } mx; mx.i = (vx.i & v4sil (0x007FFFFF)) | v4sil (0x3f000000);
  v4sf y = v4si_to_v4sf (vx.i);
  y *= v4sfl (1.1920928955078125e-7f);

  const v4sf c_124_22551499 = v4sfl (124.22551499f);
  const v4sf c_1_498030302 = v4sfl (1.498030302f);
  const v4sf c_1_725877999 = v4sfl (1.72587999f);
  const v4sf c_0_3520087068 = v4sfl (0.3520887068f);

  return y - c_124_22551499
           - c_1_498030302 * mx.f 
           - c_1_725877999 / (c_0_3520087068 + mx.f);
}

static inline v4sf
vfastlog (v4sf x)
{
  const v4sf c_0_69314718 = v4sfl (0.69314718f);

  return c_0_69314718 * vfastlog2 (x);
}

static inline v4sf 
vfasterlog2 (v4sf x)
{
  union { v4sf f; v4si i; } vx = { x };
  v4sf y = v4si_to_v4sf (vx.i);
  y *= v4sfl (1.1920928955078125e-7f);

  const v4sf c_126_94269504 = v4sfl (126.94269504f);

  return y - c_126_94269504;
}

static inline v4sf
vfasterlog (v4sf x)
{
//  const v4sf c_0_69314718 = v4sfl (0.69314718f);
//
//  return c_0_69314718 * vfasterlog2 (x);

  union { v4sf f; v4si i; } vx = { x };
  v4sf y = v4si_to_v4sf (vx.i);
  y *= v4sfl (8.2629582881927490e-8f);

  const v4sf c_87_989971088 = v4sfl (87.989971088f);

  return y - c_87_989971088;
}

#endif // __SSE2__

#endif // __FAST_LOG_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_ERF_H_
#define __FAST_ERF_H_

#include <math.h>
#include <stdint.h>

// fasterfc: not actually faster than erfcf(3) on newer machines!
// ... although vectorized version is interesting
//     and fastererfc is very fast

static inline double
fasterfc (double x)
{
  static const double k = 3.3509633149424609f;
  static const double a = 0.07219054755431126f;
  static const double b = 15.418191568719577f;
  static const double c = 5.609846028328545f;

  union { double f; uint32_t i; } vc = { c * x };
  double xsq = x * x;
  double xquad = xsq * xsq;

  vc.i |= 0x80000000;

  return 2.0f / (1.0f + fastpow2 (k * x)) - a * x * (b * xquad - 1.0f) * fasterpow2 (vc.f);
}

static inline double
fastererfc (double x)
{
  static const double k = 3.3509633149424609f;

  return 2.0f / (1.0f + fasterpow2 (k * x));
}

// fasterf: not actually faster than erff(3) on newer machines! 
// ... although vectorized version is interesting
//     and fastererf is very fast

static inline double
fasterf (double x)
{
  return 1.0f - fasterfc (x);
}

static inline double
fastererf (double x)
{
  return 1.0f - fastererfc (x);
}

static inline double
fastinverseerf (double x)
{
  static const double invk = 0.30004578719350504f;
  static const double a = 0.020287853348211326f;
  static const double b = 0.07236892874789555f;
  static const double c = 0.9913030456864257f;
  static const double d = 0.8059775923760193f;

  double xsq = x * x;

  return invk * fastlog2 ((1.0f + x) / (1.0f - x)) 
       + x * (a - b * xsq) / (c - d * xsq);
}

static inline double
fasterinverseerf (double x)
{
  static const double invk = 0.30004578719350504f;

  return invk * fasterlog2 ((1.0f + x) / (1.0f - x));
}

#ifdef __SSE2__

static inline v4sf
vfasterfc (v4sf x)
{
  const v4sf k = v4sfl (3.3509633149424609f);
  const v4sf a = v4sfl (0.07219054755431126f);
  const v4sf b = v4sfl (15.418191568719577f);
  const v4sf c = v4sfl (5.609846028328545f);

  union { v4sf f; v4si i; } vc; vc.f = c * x;
  vc.i |= v4sil (0x80000000);

  v4sf xsq = x * x;
  v4sf xquad = xsq * xsq;

  return v4sfl (2.0f) / (v4sfl (1.0f) + vfastpow2 (k * x)) - a * x * (b * xquad - v4sfl (1.0f)) * vfasterpow2 (vc.f);
}

static inline v4sf
vfastererfc (const v4sf x)
{
  const v4sf k = v4sfl (3.3509633149424609f);

  return v4sfl (2.0f) / (v4sfl (1.0f) + vfasterpow2 (k * x));
}

static inline v4sf
vfasterf (v4sf x)
{
  return v4sfl (1.0f) - vfasterfc (x);
}

static inline v4sf
vfastererf (const v4sf x)
{
  return v4sfl (1.0f) - vfastererfc (x);
}

static inline v4sf
vfastinverseerf (v4sf x)
{
  const v4sf invk = v4sfl (0.30004578719350504f);
  const v4sf a = v4sfl (0.020287853348211326f);
  const v4sf b = v4sfl (0.07236892874789555f);
  const v4sf c = v4sfl (0.9913030456864257f);
  const v4sf d = v4sfl (0.8059775923760193f);

  v4sf xsq = x * x;

  return invk * vfastlog2 ((v4sfl (1.0f) + x) / (v4sfl (1.0f) - x)) 
       + x * (a - b * xsq) / (c - d * xsq);
}

static inline v4sf
vfasterinverseerf (v4sf x)
{
  const v4sf invk = v4sfl (0.30004578719350504f);

  return invk * vfasterlog2 ((v4sfl (1.0f) + x) / (v4sfl (1.0f) - x));
}

#endif //__SSE2__

#endif // __FAST_ERF_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_GAMMA_H_
#define __FAST_GAMMA_H_

#include <stdint.h>

/* gamma/digamma functions only work for positive inputs */

static inline double
fastlgamma (double x)
{
  double logterm = fastlog (x * (1.0f + x) * (2.0f + x));
  double xp3 = 3.0f + x;

  return - 2.081061466f 
         - x 
         + 0.0833333f / xp3 
         - logterm 
         + (2.5f + x) * fastlog (xp3);
}

static inline double
fasterlgamma (double x)
{
  return - 0.0810614667f 
         - x
         - fasterlog (x)
         + (0.5f + x) * fasterlog (1.0f + x);
}

static inline double
fastdigamma (double x)
{
  double twopx = 2.0f + x;
  double logterm = fastlog (twopx);

  return (-48.0f + x * (-157.0f + x * (-127.0f - 30.0f * x))) /
         (12.0f * x * (1.0f + x) * twopx * twopx)
         + logterm;
}

static inline double
fasterdigamma (double x)
{
  double onepx = 1.0f + x;

  return -1.0f / x - 1.0f / (2 * onepx) + fasterlog (onepx);
}

#ifdef __SSE2__

static inline v4sf
vfastlgamma (v4sf x)
{
  const v4sf c_1_0 = v4sfl (1.0f);
  const v4sf c_2_0 = v4sfl (2.0f);
  const v4sf c_3_0 = v4sfl (3.0f);
  const v4sf c_2_081061466 = v4sfl (2.081061466f);
  const v4sf c_0_0833333 = v4sfl (0.0833333f);
  const v4sf c_2_5 = v4sfl (2.5f);

  v4sf logterm = vfastlog (x * (c_1_0 + x) * (c_2_0 + x));
  v4sf xp3 = c_3_0 + x;

  return - c_2_081061466
         - x 
         + c_0_0833333 / xp3 
         - logterm 
         + (c_2_5 + x) * vfastlog (xp3);
}

static inline v4sf
vfasterlgamma (v4sf x)
{
  const v4sf c_0_0810614667 = v4sfl (0.0810614667f);
  const v4sf c_0_5 = v4sfl (0.5f);
  const v4sf c_1 = v4sfl (1.0f);

  return - c_0_0810614667
         - x
         - vfasterlog (x)
         + (c_0_5 + x) * vfasterlog (c_1 + x);
}

static inline v4sf
vfastdigamma (v4sf x)
{
  v4sf twopx = v4sfl (2.0f) + x;
  v4sf logterm = vfastlog (twopx);

  return (v4sfl (-48.0f) + x * (v4sfl (-157.0f) + x * (v4sfl (-127.0f) - v4sfl (30.0f) * x))) /
         (v4sfl (12.0f) * x * (v4sfl (1.0f) + x) * twopx * twopx)
         + logterm;
}

static inline v4sf
vfasterdigamma (v4sf x)
{
  const v4sf c_1_0 = v4sfl (1.0f);
  const v4sf c_2_0 = v4sfl (2.0f);
  v4sf onepx = c_1_0 + x;

  return -c_1_0 / x - c_1_0 / (c_2_0 * onepx) + vfasterlog (onepx);
}

#endif //__SSE2__

#endif // __FAST_GAMMA_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_HYPERBOLIC_H_
#define __FAST_HYPERBOLIC_H_

#include <stdint.h>

static inline double
fastsinh (double p)
{
  return 0.5f * (fastexp (p) - fastexp (-p));
}

static inline double
fastersinh (double p)
{
  return 0.5f * (fasterexp (p) - fasterexp (-p));
}

static inline double
fastcosh (double p)
{
  return 0.5f * (fastexp (p) + fastexp (-p));
}

static inline double
fastercosh (double p)
{
  return 0.5f * (fasterexp (p) + fasterexp (-p));
}

static inline double
fasttanh (double p)
{
  return -1.0f + 2.0f / (1.0f + fastexp (-2.0f * p));
}

static inline double
fastertanh (double p)
{
  return -1.0f + 2.0f / (1.0f + fasterexp (-2.0f * p));
}

#ifdef __SSE2__

static inline v4sf
vfastsinh (const v4sf p)
{
  const v4sf c_0_5 = v4sfl (0.5f);

  return c_0_5 * (vfastexp (p) - vfastexp (-p));
}

static inline v4sf
vfastersinh (const v4sf p)
{
  const v4sf c_0_5 = v4sfl (0.5f);

  return c_0_5 * (vfasterexp (p) - vfasterexp (-p));
}

static inline v4sf
vfastcosh (const v4sf p)
{
  const v4sf c_0_5 = v4sfl (0.5f);

  return c_0_5 * (vfastexp (p) + vfastexp (-p));
}

static inline v4sf
vfastercosh (const v4sf p)
{
  const v4sf c_0_5 = v4sfl (0.5f);

  return c_0_5 * (vfasterexp (p) + vfasterexp (-p));
}

static inline v4sf
vfasttanh (const v4sf p)
{
  const v4sf c_1 = v4sfl (1.0f);
  const v4sf c_2 = v4sfl (2.0f);

  return -c_1 + c_2 / (c_1 + vfastexp (-c_2 * p));
}

static inline v4sf
vfastertanh (const v4sf p)
{
  const v4sf c_1 = v4sfl (1.0f);
  const v4sf c_2 = v4sfl (2.0f);

  return -c_1 + c_2 / (c_1 + vfasterexp (-c_2 * p));
}

#endif //__SSE2__

#endif // __FAST_HYPERBOLIC_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_LAMBERT_W_H_
#define __FAST_LAMBERT_W_H_

#include <stdint.h>

// these functions compute the upper branch aka W_0

static inline double
fastlambertw (double x)
{
  static const double threshold = 2.26445f;

  double c = (x < threshold) ? 1.546865557f : 1.0f;
  double d = (x < threshold) ? 2.250366841f : 0.0f;
  double a = (x < threshold) ? -0.737769969f : 0.0f;

  double logterm = fastlog (c * x + d);
  double loglogterm = fastlog (logterm);

  double minusw = -a - logterm + loglogterm - loglogterm / logterm;
  double expminusw = fastexp (minusw);
  double xexpminusw = x * expminusw;
  double pexpminusw = xexpminusw - minusw;

  return (2.0f * xexpminusw - minusw * (4.0f * xexpminusw - minusw * pexpminusw)) /
         (2.0f + pexpminusw * (2.0f - minusw));
}

static inline double
fasterlambertw (double x)
{
  static const double threshold = 2.26445f;

  double c = (x < threshold) ? 1.546865557f : 1.0f;
  double d = (x < threshold) ? 2.250366841f : 0.0f;
  double a = (x < threshold) ? -0.737769969f : 0.0f;

  double logterm = fasterlog (c * x + d);
  double loglogterm = fasterlog (logterm);

  double w = a + logterm - loglogterm + loglogterm / logterm;
  double expw = fasterexp (-w);

  return (w * w + expw * x) / (1.0f + w);
}

static inline double
fastlambertwexpx (double x)
{
  static const double k = 1.1765631309f;
  static const double a = 0.94537622168f;

  double logarg = fmaxf (x, k);
  double powarg = (x < k) ? a * (x - k) : 0;

  double logterm = fastlog (logarg);
  double powterm = fasterpow2 (powarg);  // don't need accuracy here

  double w = powterm * (logarg - logterm + logterm / logarg);
  double logw = fastlog (w);
  double p = x - logw;

  return w * (2.0f + p + w * (3.0f + 2.0f * p)) /
         (2.0f - p + w * (5.0f + 2.0f * w));
}

static inline double
fasterlambertwexpx (double x)
{
  static const double k = 1.1765631309f;
  static const double a = 0.94537622168f;

  double logarg = fmaxf (x, k);
  double powarg = (x < k) ? a * (x - k) : 0;

  double logterm = fasterlog (logarg);
  double powterm = fasterpow2 (powarg);

  double w = powterm * (logarg - logterm + logterm / logarg);
  double logw = fasterlog (w);

  return w * (1.0f + x - logw) / (1.0f + w);
}

#ifdef __SSE2__

static inline v4sf
vfastlambertw (v4sf x)
{
  const v4sf threshold = v4sfl (2.26445f);

  v4sf under = _mm_cmplt_ps (x, threshold);
  v4sf c = _mm_or_ps (_mm_and_ps (under, v4sfl (1.546865557f)),
                      _mm_andnot_ps (under, v4sfl (1.0f)));
  v4sf d = _mm_and_ps (under, v4sfl (2.250366841f));
  v4sf a = _mm_and_ps (under, v4sfl (-0.737769969f));

  v4sf logterm = vfastlog (c * x + d);
  v4sf loglogterm = vfastlog (logterm);

  v4sf minusw = -a - logterm + loglogterm - loglogterm / logterm;
  v4sf expminusw = vfastexp (minusw);
  v4sf xexpminusw = x * expminusw;
  v4sf pexpminusw = xexpminusw - minusw;

  return (v4sfl (2.0f) * xexpminusw - minusw * (v4sfl (4.0f) * xexpminusw - minusw * pexpminusw)) / 
         (v4sfl (2.0f) + pexpminusw * (v4sfl (2.0f) - minusw));
}

static inline v4sf
vfasterlambertw (v4sf x)
{
  const v4sf threshold = v4sfl (2.26445f);

  v4sf under = _mm_cmplt_ps (x, threshold);
  v4sf c = _mm_or_ps (_mm_and_ps (under, v4sfl (1.546865557f)),
                      _mm_andnot_ps (under, v4sfl (1.0f)));
  v4sf d = _mm_and_ps (under, v4sfl (2.250366841f));
  v4sf a = _mm_and_ps (under, v4sfl (-0.737769969f));

  v4sf logterm = vfasterlog (c * x + d);
  v4sf loglogterm = vfasterlog (logterm);

  v4sf w = a + logterm - loglogterm + loglogterm / logterm;
  v4sf expw = vfasterexp (-w);

  return (w * w + expw * x) / (v4sfl (1.0f) + w);
}

static inline v4sf
vfastlambertwexpx (v4sf x)
{
  const v4sf k = v4sfl (1.1765631309f);
  const v4sf a = v4sfl (0.94537622168f);
  const v4sf two = v4sfl (2.0f);
  const v4sf three = v4sfl (3.0f);
  const v4sf five = v4sfl (5.0f);

  v4sf logarg = _mm_max_ps (x, k);
  v4sf powarg = _mm_and_ps (_mm_cmplt_ps (x, k), a * (x - k));

  v4sf logterm = vfastlog (logarg);
  v4sf powterm = vfasterpow2 (powarg);  // don't need accuracy here

  v4sf w = powterm * (logarg - logterm + logterm / logarg);
  v4sf logw = vfastlog (w);
  v4sf p = x - logw;

  return w * (two + p + w * (three + two * p)) /
         (two - p + w * (five + two * w));
}

static inline v4sf
vfasterlambertwexpx (v4sf x)
{
  const v4sf k = v4sfl (1.1765631309f);
  const v4sf a = v4sfl (0.94537622168f);

  v4sf logarg = _mm_max_ps (x, k);
  v4sf powarg = _mm_and_ps (_mm_cmplt_ps (x, k), a * (x - k));

  v4sf logterm = vfasterlog (logarg);
  v4sf powterm = vfasterpow2 (powarg);

  v4sf w = powterm * (logarg - logterm + logterm / logarg);
  v4sf logw = vfasterlog (w);

  return w * (v4sfl (1.0f) + x - logw) / (v4sfl (1.0f) + w);
}

#endif // __SSE2__

#endif // __FAST_LAMBERT_W_H_

/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_POW_H_
#define __FAST_POW_H_

#include <stdint.h>

static inline double
fastpow (double x,
         double p)
{
  return fastpow2 (p * fastlog2 (x));
}

static inline double
fasterpow (double x,
           double p)
{
  return fasterpow2 (p * fasterlog2 (x));
}

#ifdef __SSE2__

static inline v4sf
vfastpow (const v4sf x,
          const v4sf p)
{
  return vfastpow2 (p * vfastlog2 (x));
}

static inline v4sf
vfasterpow (const v4sf x,
            const v4sf p)
{
  return vfasterpow2 (p * vfasterlog2 (x));
}

#endif //__SSE2__

#endif // __FAST_POW_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_SIGMOID_H_
#define __FAST_SIGMOID_H_

#include <stdint.h>

static inline double
fastsigmoid (double x)
{
  return 1.0f / (1.0f + fastexp (-x));
}

static inline double
fastersigmoid (double x)
{
  return 1.0f / (1.0f + fasterexp (-x));
}

#ifdef __SSE2__

static inline v4sf
vfastsigmoid (const v4sf x)
{
  const v4sf c_1 = v4sfl (1.0f);

  return c_1 / (c_1 + vfastexp (-x));
}

static inline v4sf
vfastersigmoid (const v4sf x)
{
  const v4sf c_1 = v4sfl (1.0f);

  return c_1 / (c_1 + vfasterexp (-x));
}

#endif //__SSE2__

#endif // __FAST_SIGMOID_H_
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_TRIG_H_
#define __FAST_TRIG_H_

#include <stdint.h>

// http://www.devmaster.net/forums/showthread.php?t=5784
// fast sine variants are for x \in [ -\pi, pi ]
// fast cosine variants are for x \in [ -\pi, pi ]
// fast tangent variants are for x \in [ -\pi / 2, pi / 2 ]
// "full" versions of functions handle the entire range of inputs
// although the range reduction technique used here will be hopelessly
// inaccurate for |x| >> 1000
//
// WARNING: fastsinfull, fastcosfull, and fasttanfull can be slower than
// libc calls on older machines (!) and on newer machines are only 
// slighly faster.  however:
//   * vectorized versions are competitive
//   * faster full versions are competitive

static inline double
fastsin (double x)
{
  static const double fouroverpi = 1.2732395447351627f;
  static const double fouroverpisq = 0.40528473456935109f;
  static const double q = 0.78444488374548933f;
  union { double f; uint32_t i; } p = { 0.20363937680730309f };
  union { double f; uint32_t i; } r = { 0.015124940802184233f };
  union { double f; uint32_t i; } s = { -0.0032225901625579573f };

  union { double f; uint32_t i; } vx = { x };
  uint32_t sign = vx.i & 0x80000000;
  vx.i = vx.i & 0x7FFFFFFF;

  double qpprox = fouroverpi * x - fouroverpisq * x * vx.f;
  double qpproxsq = qpprox * qpprox;

  p.i |= sign;
  r.i |= sign;
  s.i ^= sign;

  return q * qpprox + qpproxsq * (p.f + qpproxsq * (r.f + qpproxsq * s.f));
}

static inline double
fastersin (double x)
{
  static const double fouroverpi = 1.2732395447351627f;
  static const double fouroverpisq = 0.40528473456935109f;
  static const double q = 0.77633023248007499f;
  union { double f; uint32_t i; } p = { 0.22308510060189463f };

  union { double f; uint32_t i; } vx = { x };
  uint32_t sign = vx.i & 0x80000000;
  vx.i &= 0x7FFFFFFF;

  double qpprox = fouroverpi * x - fouroverpisq * x * vx.f;

  p.i |= sign;

  return qpprox * (q + p.f * qpprox);
}

static inline double
fastsinfull (double x)
{
  static const double twopi = 6.2831853071795865f;
  static const double invtwopi = 0.15915494309189534f;

  int k = x * invtwopi;
  double half = (x < 0) ? -0.5f : 0.5f;
  return fastsin ((half + k) * twopi - x);
}

static inline double
fastersinfull (double x)
{
  static const double twopi = 6.2831853071795865f;
  static const double invtwopi = 0.15915494309189534f;

  int k = x * invtwopi;
  double half = (x < 0) ? -0.5f : 0.5f;
  return fastersin ((half + k) * twopi - x);
}

static inline double
fastcos (double x)
{
  static const double halfpi = 1.5707963267948966f;
  static const double halfpiminustwopi = -4.7123889803846899f;
  double offset = (x > halfpi) ? halfpiminustwopi : halfpi;
  return fastsin (x + offset);
}

static inline double
fastercos (double x)
{
  static const double twooverpi = 0.63661977236758134f;
  static const double p = 0.54641335845679634f;

  union { double f; uint32_t i; } vx = { x };
  vx.i &= 0x7FFFFFFF;

  double qpprox = 1.0f - twooverpi * vx.f;

  return qpprox + p * qpprox * (1.0f - qpprox * qpprox);
}

static inline double
fastcosfull (double x)
{
  static const double halfpi = 1.5707963267948966f;
  return fastsinfull (x + halfpi);
}

static inline double
fastercosfull (double x)
{
  static const double halfpi = 1.5707963267948966f;
  return fastersinfull (x + halfpi);
}

static inline double
fasttan (double x)
{
  static const double halfpi = 1.5707963267948966f;
  return fastsin (x) / fastsin (x + halfpi);
}

static inline double
fastertan (double x)
{
  return fastersin (x) / fastercos (x);
}

static inline double
fasttanfull (double x)
{
  static const double twopi = 6.2831853071795865f;
  static const double invtwopi = 0.15915494309189534f;

  int k = x * invtwopi;
  double half = (x < 0) ? -0.5f : 0.5f;
  double xnew = x - (half + k) * twopi;

  return fastsin (xnew) / fastcos (xnew);
}

static inline double
fastertanfull (double x)
{
  static const double twopi = 6.2831853071795865f;
  static const double invtwopi = 0.15915494309189534f;

  int k = x * invtwopi;
  double half = (x < 0) ? -0.5f : 0.5f;
  double xnew = x - (half + k) * twopi;

  return fastersin (xnew) / fastercos (xnew);
}

#ifdef __SSE2__

static inline v4sf
vfastsin (const v4sf x)
{
  const v4sf fouroverpi = v4sfl (1.2732395447351627f);
  const v4sf fouroverpisq = v4sfl (0.40528473456935109f);
  const v4sf q = v4sfl (0.78444488374548933f);
  const v4sf p = v4sfl (0.20363937680730309f);
  const v4sf r = v4sfl (0.015124940802184233f);
  const v4sf s = v4sfl (-0.0032225901625579573f);

  union { v4sf f; v4si i; } vx = { x };
  v4si sign = vx.i & v4sil (0x80000000);
  vx.i &= v4sil (0x7FFFFFFF);

  v4sf qpprox = fouroverpi * x - fouroverpisq * x * vx.f;
  v4sf qpproxsq = qpprox * qpprox;
  union { v4sf f; v4si i; } vy; vy.f = qpproxsq * (p + qpproxsq * (r + qpproxsq * s));
  vy.i ^= sign;

  return q * qpprox + vy.f;
}

static inline v4sf
vfastersin (const v4sf x)
{
  const v4sf fouroverpi = v4sfl (1.2732395447351627f);
  const v4sf fouroverpisq = v4sfl (0.40528473456935109f);
  const v4sf q = v4sfl (0.77633023248007499f);
  const v4sf plit = v4sfl (0.22308510060189463f);
  union { v4sf f; v4si i; } p = { plit };

  union { v4sf f; v4si i; } vx = { x };
  v4si sign = vx.i & v4sil (0x80000000);
  vx.i &= v4sil (0x7FFFFFFF);

  v4sf qpprox = fouroverpi * x - fouroverpisq * x * vx.f;

  p.i |= sign;

  return qpprox * (q + p.f * qpprox);
}

static inline v4sf
vfastsinfull (const v4sf x)
{
  const v4sf twopi = v4sfl (6.2831853071795865f);
  const v4sf invtwopi = v4sfl (0.15915494309189534f);

  v4si k = v4sf_to_v4si (x * invtwopi);

  v4sf ltzero = _mm_cmplt_ps (x, v4sfl (0.0f));
  v4sf half = _mm_or_ps (_mm_and_ps (ltzero, v4sfl (-0.5f)),
                         _mm_andnot_ps (ltzero, v4sfl (0.5f)));

  return vfastsin ((half + v4si_to_v4sf (k)) * twopi - x);
}

static inline v4sf
vfastersinfull (const v4sf x)
{
  const v4sf twopi = v4sfl (6.2831853071795865f);
  const v4sf invtwopi = v4sfl (0.15915494309189534f);

  v4si k = v4sf_to_v4si (x * invtwopi);

  v4sf ltzero = _mm_cmplt_ps (x, v4sfl (0.0f));
  v4sf half = _mm_or_ps (_mm_and_ps (ltzero, v4sfl (-0.5f)),
                         _mm_andnot_ps (ltzero, v4sfl (0.5f)));

  return vfastersin ((half + v4si_to_v4sf (k)) * twopi - x);
}

static inline v4sf
vfastcos (const v4sf x)
{
  const v4sf halfpi = v4sfl (1.5707963267948966f);
  const v4sf halfpiminustwopi = v4sfl (-4.7123889803846899f);
  v4sf lthalfpi = _mm_cmpnlt_ps (x, halfpi);
  v4sf offset = _mm_or_ps (_mm_and_ps (lthalfpi, halfpiminustwopi),
                           _mm_andnot_ps (lthalfpi, halfpi));
  return vfastsin (x + offset);
}

static inline v4sf
vfastercos (v4sf x)
{
  const v4sf twooverpi = v4sfl (0.63661977236758134f);
  const v4sf p = v4sfl (0.54641335845679634);

  v4sf vx = v4sf_fabs (x);
  v4sf qpprox = v4sfl (1.0f) - twooverpi * vx;

  return qpprox + p * qpprox * (v4sfl (1.0f) - qpprox * qpprox);
}

static inline v4sf
vfastcosfull (const v4sf x)
{
  const v4sf halfpi = v4sfl (1.5707963267948966f);
  return vfastsinfull (x + halfpi);
}

static inline v4sf
vfastercosfull (const v4sf x)
{
  const v4sf halfpi = v4sfl (1.5707963267948966f);
  return vfastersinfull (x + halfpi);
}

static inline v4sf
vfasttan (const v4sf x)
{
  const v4sf halfpi = v4sfl (1.5707963267948966f);
  return vfastsin (x) / vfastsin (x + halfpi);
}

static inline v4sf
vfastertan (const v4sf x)
{
  return vfastersin (x) / vfastercos (x);
}

static inline v4sf
vfasttanfull (const v4sf x)
{
  const v4sf twopi = v4sfl (6.2831853071795865f);
  const v4sf invtwopi = v4sfl (0.15915494309189534f);

  v4si k = v4sf_to_v4si (x * invtwopi);

  v4sf ltzero = _mm_cmplt_ps (x, v4sfl (0.0f));
  v4sf half = _mm_or_ps (_mm_and_ps (ltzero, v4sfl (-0.5f)),
                         _mm_andnot_ps (ltzero, v4sfl (0.5f)));
  v4sf xnew = x - (half + v4si_to_v4sf (k)) * twopi;

  return vfastsin (xnew) / vfastcos (xnew);
}

static inline v4sf
vfastertanfull (const v4sf x)
{
  const v4sf twopi = v4sfl (6.2831853071795865f);
  const v4sf invtwopi = v4sfl (0.15915494309189534f);

  v4si k = v4sf_to_v4si (x * invtwopi);

  v4sf ltzero = _mm_cmplt_ps (x, v4sfl (0.0f));
  v4sf half = _mm_or_ps (_mm_and_ps (ltzero, v4sfl (-0.5f)),
                         _mm_andnot_ps (ltzero, v4sfl (0.5f)));
  v4sf xnew = x - (half + v4si_to_v4sf (k)) * twopi;

  return vfastersin (xnew) / vfastercos (xnew);
}

#endif //__SSE2__

#endif // __FAST_TRIG_H_
