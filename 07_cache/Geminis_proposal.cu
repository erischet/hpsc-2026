#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
#include <cuda_fp16.h>

using namespace std;
using namespace nvcuda;

__device__ __forceinline__ void get_swizzled_block(int &bx, int &by, int grid_width, int grid_height) {
  const int PANEL = 8;
  int block_id = blockIdx.y * grid_width + blockIdx.x;
  int panel_id = block_id / (PANEL * grid_height);
  int panel_offset = block_id % (PANEL * grid_height);
  bx = panel_id * PANEL + (panel_offset % PANEL);
  by = panel_offset / PANEL;
  
  if (bx >= grid_width) {
    bx = blockIdx.x;
    by = blockIdx.y;
  }
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
                       float *d_a, float *d_b, float *d_c) {
  
  int bx, by;
  get_swizzled_block(bx, by, gridDim.x, gridDim.y);

  int offset_a_m = 128 * bx;
  int offset_b_n = 128 * by;
  int tid = threadIdx.x;
  
  int warp_id = tid / 32;
  int warp_m = warp_id % 4;
  int warp_n = warp_id / 4;

  extern __shared__ half smem[];
  half (*smem_A)[32][136] = reinterpret_cast<half (*)[32][136]>(smem);
  half (*smem_B)[128][40] = reinterpret_cast<half (*)[128][40]>(smem + (3 * 32 * 136));

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  #pragma unroll
  for (int i = 0; i < 2; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

  #define LOAD_STAGE(STAGE, K_OFFSET) \
  do { \
    if ((K_OFFSET) < dim_k) { \
      _Pragma("unroll") \
      for (int step = 0; step < 4; ++step) { \
        int k_a = (tid / 32) + step * 8; \
        int m_a = (tid % 32) * 4; \
        float4 tmp_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f); \
        if ((K_OFFSET + k_a) < dim_k && (offset_a_m + m_a) < dim_m) { \
          tmp_a = reinterpret_cast<const float4*>(&d_a[(K_OFFSET + k_a) * dim_m + offset_a_m + m_a])[0]; \
        } \
        reinterpret_cast<half2*>(&smem_A[STAGE][k_a][m_a])[0] = __floats2half2_rn(tmp_a.x, tmp_a.y); \
        reinterpret_cast<half2*>(&smem_A[STAGE][k_a][m_a])[1] = __floats2half2_rn(tmp_a.z, tmp_a.w); \
      } \
      _Pragma("unroll") \
      for (int step = 0; step < 4; ++step) { \
        int n_b = (tid / 8) + step * 32; \
        int k_b = (tid % 8) * 4; \
        float4 tmp_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f); \
        if ((offset_b_n + n_b) < dim_n && (K_OFFSET + k_b) < dim_k) { \
          tmp_b = reinterpret_cast<const float4*>(&d_b[(offset_b_n + n_b) * dim_k + K_OFFSET + k_b])[0]; \
        } \
        reinterpret_cast<half2*>(&smem_B[STAGE][n_b][k_b])[0] = __floats2half2_rn(tmp_b.x, tmp_b.y); \
        reinterpret_cast<half2*>(&smem_B[STAGE][n_b][k_b])[1] = __floats2half2_rn(tmp_b.z, tmp_b.w); \
      } \
    } \
  } while(0)

  LOAD_STAGE(0, 0);
  LOAD_STAGE(1, 32);
  __syncthreads();

  for (int k_idx = 0; k_idx < dim_k; k_idx += 32) {
    int next_k = k_idx + 64; 
    int write_stage = (k_idx / 32 + 2) % 3;
    int read_stage = (k_idx / 32) % 3;

    LOAD_STAGE(write_stage, next_k);

    #pragma unroll
    for (int k_step = 0; k_step < 32; k_step += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];

      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        wmma::load_matrix_sync(a_frag[i], &smem_A[read_stage][k_step][warp_m * 32 + i * 16], 136);
      }
      
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        wmma::load_matrix_sync(b_frag[j], &smem_B[read_stage][warp_n * 64 + j * 16][k_step], 40);
      }

      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c_m = offset_a_m + warp_m * 32 + i * 16;
      int c_n = offset_b_n + warp_n * 64 + j * 16;
      if (c_n < dim_n && c_m < dim_m) {
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[i][j], dim_m, wmma::mem_col_major);
      }
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
      
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
      
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
      
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
         CUBLAS_OP_N,
         CUBLAS_OP_N,
         m, n, k,
         &alpha,
         A, CUDA_R_32F, m,
         B, CUDA_R_32F, k,
         &beta,
         C, CUDA_R_32F, m,
         CUBLAS_COMPUTE_32F_FAST_16F,
         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  
  int tile = 128;
  int threads = 256; 
  dim3 block = dim3(threads);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);

  int smem_size = (3 * 32 * 136 + 3 * 128 * 40) * sizeof(half);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block, smem_size >>>(m, n, k, A, B, C2);
    cudaDeviceSynchronize();
  }
  
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  
  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C2);
  cublasDestroy(cublas_handle);
}