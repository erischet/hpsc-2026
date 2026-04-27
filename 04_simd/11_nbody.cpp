#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h> 

int main() {
  //CPU doesn't support AVX-512 --> switch to AVX2
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 zerovec = _mm256_setzero_ps();
  __m256 vecx_masked = _mm256_setzero_ps();
  __m256 vecy_masked = _mm256_setzero_ps();

for(int i = 0; i < N; i++) {
        __m256 x_target_vec = _mm256_set1_ps(x[i]);
        __m256 y_target_vec = _mm256_set1_ps(y[i]);

        __m256 rx = _mm256_sub_ps(x_target_vec, xvec);
        __m256 ry = _mm256_sub_ps(y_target_vec, yvec);

        __m256 r2 = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry));

        __m256 mask = _mm256_cmp_ps(r2, _mm256_setzero_ps(), _CMP_GT_OQ);

        __m256 inv_r = _mm256_rsqrt_ps(r2);
        __m256 inv_r2 = _mm256_mul_ps(inv_r, inv_r);
        __m256 inv_r3 = _mm256_mul_ps(inv_r, inv_r2);

        __m256 fx_v = _mm256_mul_ps(_mm256_mul_ps(rx, mvec), inv_r3);
        __m256 fy_v = _mm256_mul_ps(_mm256_mul_ps(ry, mvec), inv_r3);

        fx_v = _mm256_and_ps(fx_v, mask);
        fy_v = _mm256_and_ps(fy_v, mask);

        alignas(32) float temp_x[8], temp_y[8];
        _mm256_store_ps(temp_x, fx_v);
        _mm256_store_ps(temp_y, fy_v);

        for(int j = 0; j < N; j++) {
            fx[i] -= temp_x[j];
            fy[i] -= temp_y[j];
        }

        printf("%d %g %g\n", i, fx[i], fy[i]);
    }
    //compared - gives same results as original algorithm with N = 8
  }


