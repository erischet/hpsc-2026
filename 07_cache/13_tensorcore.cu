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

// Asynchroner Kopiervorgang (16 Bytes / 8 Halfs)
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* global_ptr) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(smem_addr), "l"(global_ptr)
  );
}

// Typen von d_a und d_b müssen für cp.async zwingend half* sein
__global__ void kernel(int dim_m, int dim_n, int dim_k,
                       half *d_a, half *d_b, float *d_c) {

  // Grid Swizzling: Umverteilung der Blöcke in 8er-Panels
  const int panel_width = 8;
  int bid = blockIdx.y * gridDim.x + blockIdx.x; 
  
  int panel_id = bid / (gridDim.y * panel_width);
  int bid_within_panel = bid % (gridDim.y * panel_width);
  
  int new_blockIdx_x = panel_id * panel_width + (bid_within_panel % panel_width);
  int new_blockIdx_y = bid_within_panel / panel_width;
  
  if (new_blockIdx_x >= gridDim.x || new_blockIdx_y >= gridDim.y) return;
 
  // Offset angepasst auf Kachel 128x128
  int offset_a_m = 128 * new_blockIdx_x;
  int offset_b_n = 128 * new_blockIdx_y;
  int tid = threadIdx.x;
  
  int warp_id = tid / 32;
  int warp_m = warp_id % 4;
  int warp_n = warp_id / 4;

  // Dynamischer Shared Memory (Bypass 48-KB-Limit)
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

  // Lade-Makro über 256 Threads verteilt (Vermeidet Register-Engpass)
  #define LOAD_STAGE(STAGE, K_OFFSET) \
  do { \
    if ((K_OFFSET) < dim_k) { \
      _Pragma("unroll") \
      for (int step = 0; step < 2; ++step) { \
        int logical_id = step * 256 + tid; \
        int k_a = logical_id / 16; \
        int m_a = (logical_id % 16) * 8; \
        cp_async_16B(&smem_A[STAGE][k_a][m_a], &d_a[(K_OFFSET + k_a) * dim_m + offset_a_m + m_a]); \
      } \
      _Pragma("unroll") \
      for (int step = 0; step < 2; ++step) { \
        int logical_id = step * 256 + tid; \
        int n_b = logical_id / 4; \
        int k_b = (logical_id % 4) * 8; \
        cp_async_16B(&smem_B[STAGE][n_b][k_b], &d_b[(offset_b_n + n_b) * dim_k + (K_OFFSET + k_b)]); \
      } \
    } \
  } while(0)

  // Prolog: Erste 2 Stufen laden
  LOAD_STAGE(0, 0);
  asm volatile("cp.async.commit_group;\n" ::);
  LOAD_STAGE(1, 32);
  asm volatile("cp.async.commit_group;\n" ::);
  
  // Warten bis max 1 Gruppe in Bearbeitung
  asm volatile("cp.async.wait_group 1;\n" ::);
  __syncthreads();

  // 3-Stufen-Pipeline Schleife
  for (int k_idx = 0; k_idx < dim_k; k_idx += 32) {
    int next_k = k_idx + 64; 
    int write_stage = (k_idx / 32 + 2) % 3;
    int read_stage = (k_idx / 32) % 3;

    LOAD_STAGE(write_stage, next_k);
    asm volatile("cp.async.commit_group;\n" ::);

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
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
  }

  // Epilog
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
  
  // A und B als half deklarieren
  half *A, *B;
  float *C, *C2;
  
  cudaMallocManaged(&A, m * k * sizeof(half));
  cudaMallocManaged(&B, k * n * sizeof(half));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  
  // Matrix A und B initialisieren und konvertieren
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = __float2half((float)drand48());
      
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = __float2half((float)drand48());
      
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
         A, CUDA_R_16F, m, // Umgestellt auf 16F
         B, CUDA_R_16F, k, // Umgestellt auf 16F
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

  // Eigene Implementierung
  int tile_m = 128, tile_n = 128;
  dim3 block(256); // 256 Threads
  dim3 grid((m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n);

  int smem_size = (3 * 32 * 136 + 3 * 128 * 40) * sizeof(half); // ca. 56.8 KB
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  for (int i = 0; i < Nt+2; i++) { // Schleife korrigiert
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<<grid, block, smem_size>>>(m, n, k, A, B, C2);
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
 * Reason: Total loss of latency hiding. Tensor Cores stalled during memory fetches.
 *
 * 12. L2-Cache Optimization (Grid Swizzling): 
 * Reordered block execution into 8x8 panels to maximize L2-Cache hits 
 * and data locality for matrices A and B.
 * Performance: ~71,412 GFLOPS.
 *
 * 13. Async Copy (cp.async) & 3-Stage Pipeline & 256 Threads: 
 * Changed A/B arrays to half*, integrated hardware-accelerated memory copies (cp.async).
 * Lifted register limit via dynamic Shared Memory (~56.8 KB) and Warp-Partitioning, 
 * successfully scaling to 256 Threads (128x128 Tile) using a 3-Stage Software Pipeline.
 * Performance: [Bitte aktuelle GFLOPS eintragen]
 */