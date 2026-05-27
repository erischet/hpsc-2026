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

__global__ void kernel(int dim_m, int dim_n, int dim_k,
           float *d_a, float *d_b, float *d_c) {

  const int panel_width = 8;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;
  
  int panel_id = bid / (gridDim.y * panel_width);
  int bid_within_panel = bid % (gridDim.y * panel_width);
  
  int new_blockIdx_x = panel_id * panel_width + (bid_within_panel % panel_width);
  int new_blockIdx_y = bid_within_panel / panel_width;
  
  if (new_blockIdx_x >= gridDim.x || new_blockIdx_y >= gridDim.y) return;

  int offset_a_m = 128 * new_blockIdx_x;
  int offset_b_n = 128 * new_blockIdx_y;
  int warp_id = threadIdx.x / 32;
  int warp_row = warp_id % 4;
  int warp_col = warp_id / 4;

  extern __shared__ half smem[];
  half (*block_a)[32][136] = reinterpret_cast<half (*)[32][136]>(smem);
  half (*block_b)[128][40] = reinterpret_cast<half (*)[128][40]>(smem + (2 * 32 * 136));

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];

  for (int r = 0; r < 2; r++)
 
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  float4 vec_a_reg[4];
  float4 vec_b_reg[4];

  // Prologue: Load Stage 0 (k = 0)

  for (int step = 0; step < 4; ++step) {
    int logical_id = step * 256 + threadIdx.x;
    int r = logical_id / 32;
    int c = (logical_id % 32) * 4;
    vec_a_reg[step] = reinterpret_cast<float4*>(&d_a[(0 + r) * dim_m + offset_a_m + c])[0];
  }

  for (int step = 0; step < 4; ++step) {
    int logical_id = step * 256 + threadIdx.x;
    int n_idx = logical_id / 8;
    int k_idx = (logical_id % 8) * 4;
    vec_b_reg[step] = reinterpret_cast<float4*>(&d_b[(offset_b_n + n_idx) * dim_k + 0 + k_idx])[0];
  }

  for (int step = 0; step < 4; ++step) {
    int logical_id = step * 256 + threadIdx.x;
    int r = logical_id / 32;
    int c = (logical_id % 32) * 4;
    reinterpret_cast<half2*>(&block_a[0][r][c])[0] = __floats2half2_rn(vec_a_reg[step].x, vec_a_reg[step].y);
    reinterpret_cast<half2*>(&block_a[0][r][c])[1] = __floats2half2_rn(vec_a_reg[step].z, vec_a_reg[step].w);
    
    int n_idx = logical_id / 8;
    int k_idx = (logical_id % 8) * 4;
    reinterpret_cast<half2*>(&block_b[0][n_idx][k_idx])[0] = __floats2half2_rn(vec_b_reg[step].x, vec_b_reg[step].y);
    reinterpret_cast<half2*>(&block_b[0][n_idx][k_idx])[1] = __floats2half2_rn(vec_b_reg[step].z, vec_b_reg[step].w);
  }

  // Prologue: Pre-Fetch Stage 1 (k = 32) in Register
  if (32 < dim_k) {

    for (int step = 0; step < 4; ++step) {
      int logical_id = step * 256 + threadIdx.x;
      int r = logical_id / 32;
      int c = (logical_id % 32) * 4;
      vec_a_reg[step] = reinterpret_cast<float4*>(&d_a[(32 + r) * dim_m + offset_a_m + c])[0];
    }

    for (int step = 0; step < 4; ++step) {
      int logical_id = step * 256 + threadIdx.x;
      int n_idx = logical_id / 8;
      int k_idx = (logical_id % 8) * 4;
      vec_b_reg[step] = reinterpret_cast<float4*>(&d_b[(offset_b_n + n_idx) * dim_k + 32 + k_idx])[0];
    }
  }
  
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];

  // Main Loop
  for (int k = 0; k < dim_k; k += 32) {
    int next_k = k + 32;   // In den Registern bereitstehende Daten
    int fetch_k = k + 64;  // Als Nächstes abzurufende Daten
    int write_idx = ((k / 32) + 1) % 2;
    int read_idx = (k / 32) % 2;

    // 1. Speichern der vorab geladenen Register im Shared Memory
    if (next_k < dim_k) {
       
      for (int step = 0; step < 4; ++step) {
        int logical_id = step * 256 + threadIdx.x;
        int r = logical_id / 32;
        int c = (logical_id % 32) * 4;
        reinterpret_cast<half2*>(&block_a[write_idx][r][c])[0] = __floats2half2_rn(vec_a_reg[step].x, vec_a_reg[step].y);
        reinterpret_cast<half2*>(&block_a[write_idx][r][c])[1] = __floats2half2_rn(vec_a_reg[step].z, vec_a_reg[step].w);
        
        int n_idx = logical_id / 8;
        int k_idx = (logical_id % 8) * 4;
        reinterpret_cast<half2*>(&block_b[write_idx][n_idx][k_idx])[0] = __floats2half2_rn(vec_b_reg[step].x, vec_b_reg[step].y);
        reinterpret_cast<half2*>(&block_b[write_idx][n_idx][k_idx])[1] = __floats2half2_rn(vec_b_reg[step].z, vec_b_reg[step].w);
      }
    }

    // 2. Initialisierung asynchroner globaler Ladevorgänge
    if (fetch_k < dim_k) {
       
      for (int step = 0; step < 4; ++step) {
        int logical_id = step * 256 + threadIdx.x;
        int r = logical_id / 32;
        int c = (logical_id % 32) * 4;
        vec_a_reg[step] = reinterpret_cast<float4*>(&d_a[(fetch_k + r) * dim_m + offset_a_m + c])[0];
      }
      
       
      for (int step = 0; step < 4; ++step) {
        int logical_id = step * 256 + threadIdx.x;
        int n_idx = logical_id / 8;
        int k_idx = (logical_id % 8) * 4;
        vec_b_reg[step] = reinterpret_cast<float4*>(&d_b[(offset_b_n + n_idx) * dim_k + fetch_k + k_idx])[0];
      }
    }

    // 3. Tensor Core Berechnungsphase
     
    for (int k_step = 0; k_step < 32; k_step += 16) {
       
      for (int r = 0; r < 2; r++) {
        int row_tile = warp_row * 2 + r;
        wmma::load_matrix_sync(a_frag[r], &block_a[read_idx][k_step][row_tile * 16], 136);
      }
       
      for (int c = 0; c < 4; c++) {
        int col_tile = warp_col * 4 + c;
        wmma::load_matrix_sync(b_frag[c], &block_b[read_idx][col_tile * 16][k_step], 40);
      }
       
      for (int r = 0; r < 2; r++) {
         
        for (int c = 0; c < 4; c++) {
          wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
        }
      }
    }
    __syncthreads();
  }

  // Epilogue: Shared Memory Staging for Coalesced Writes
  __syncthreads();
  float *smem_c = reinterpret_cast<float*>(smem);

   
  for (int r = 0; r < 2; r++) {
     
    for (int c = 0; c < 4; c++) {
      int smem_m = (warp_row * 2 + r) * 16;
      int smem_n = (warp_col * 4 + c) * 16;
      wmma::store_matrix_sync(&smem_c[smem_n * 128 + smem_m], acc[r][c], 128, wmma::mem_col_major);
    }
  }

  __syncthreads();

   
  for (int i = 0; i < 16; ++i) {
    int logical_id = i * 256 + threadIdx.x;
    int n_idx = logical_id / 32;
    int m_idx_vec = logical_id % 32;
    
    int g_m = offset_a_m + (m_idx_vec * 4);
    int g_n = offset_b_n + n_idx;

    if (g_n < dim_n && g_m < dim_m) {
      float4 out_vec = reinterpret_cast<float4*>(&smem_c[n_idx * 128 + m_idx_vec * 4])[0];
      reinterpret_cast<float4*>(&d_c[g_n * dim_m + g_m])[0] = out_vec;
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

  int tile_m = 128;
  int tile_n = 128;
  dim3 block = dim3(256);
  dim3 grid = dim3((m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n);
 
  int smem_compute = (2 * 32 * 136 + 2 * 128 * 40) * sizeof(half);
  int smem_epilogue = 128 * 128 * sizeof(float);
  int smem_size = (smem_compute > smem_epilogue) ? smem_compute : smem_epilogue;
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

/*
 * Performance Optimization History:
 * Testing Environment: gpu_h
 * Baseline Performance: ~9,517 GFLOPS
 *
 * 1. 08_block_8x8.cu: 
 * Implemented an 8x8 block_c.
 * Performance: Minimal improvement over baseline.
 *
 * 2. 09_reg_load.cu: 
 * Used vec_t (vectorized register loads).
 * Performance: ~22,000 GFLOPS.
 *
 * 3. 10_align.cu: 
 * Added __align__ to shared memory.
 * Performance: No significant improvement.
 *
 * 4. Bank Conflict Resolution: 
 * Added +8 padding to shared memory.
 * Performance: ~44,030 GFLOPS (almost 100% improvement from step 2).
 *
 * 5. Thread Scaling (128 Threads): 
 * Increased block size to 128 threads (4 warps), Tile 128x64.
 * Performance: ~66,123 GFLOPS.
 *
 * 6. Thread Scaling (256 Threads): 
 * Increased to 256 threads (8 warps), Tile 128x128.
 * Performance: Dropped to ~52,030 GFLOPS due to register spilling.
 *
 * 7. Rollback: 
 * Reverted to 128 threads / 128x64 Tile for optimal SM occupancy.
 * Performance: Restored ~66,123 GFLOPS.
 *
 * 8. Double Buffering: 
 * Implemented ping-pong buffers to hide global memory latency.
 * Performance: ~67,551 GFLOPS.
 *
 * 9. K-Dimension Scaling (K=32): 
 * Increased K-Tile size from 16 to 32.
 * Doubled arithmetic intensity by feeding 32 elements to Tensor Cores per loop.
 * Retained 128 threads and optimal register count.
 * Performance:  58874.55 Gflops
 *
 * 10. Rollback to optimal config:
 * Reverted to K=16 with Double Buffering.
 *
 * 11. K-Dimension without Double Buffering: 
 * Tested K=32 with Single Buffering.
 * Performance: Dropped to ~34,269 GFLOPS.
 * Reason: Total loss of latency hiding. Tensor Csores stalled during memory fetches.
 *
 * 12. Rollback to optimal config:
 * Reverted to 128 Threads, Tile 128x64, K=16 with Double Buffering.
 * Performance restored to: ~67,551 GFLOPS.
 *
 * 13. 3-Stage Software Pipeline: 
 * Utilize 3 stage buffering (instead of Double Buffering)
 */
 /*
 * 14. 3-Stage Software Pipeline Analysis:
 * Implemented 3-stage buffering to further hide global memory latency.
 * Result: Performance remained stagnant, with a slight decrease compared to the previous double-buffering baseline. Maybe go back later to 2-stage.
 /*
 * 15. Shared Memory Transposition & Vectorization (Matrix B):
 * Transposed Matrix B in shared memory to [128][24] (N=128, K=16 + 8 padding).
 * Replaced individual 16-bit __float2half stores with 32-bit __floats2half2_rn.
 * Changed Matrix B wmma fragment to col_major. 
 * Updated to 32 --> better than 64 (performance decrease) --> 80782.16 Gflops
 * Tried 2 stage buffering: 79901.31 Gflops
 * Tried 4 stage buffering: 80056.43 Gflops
 /*
 * 16. Register Double-Buffering & Epilogue Coalescing:
 * - Implemented register double-buffering for global loads to resolve RAW stalls and overlap memory fetches with Tensor Core computation.
 * - Added shared memory staging in the epilogue for coalesced float4 writes to global memory. --> 87430.32 Gflops
 */
  /*
 * 17. back to 2-stage buffering --> 88720.19 Gflops
 */

