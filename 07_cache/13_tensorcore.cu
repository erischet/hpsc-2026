#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

__global__ void kernel(int dim_m, int dim_n, int dim_k,
           float *d_a, float *d_b, float *d_c) {
  int offset_a_m = 128 * blockIdx.x;
  int offset_b_n = 64 * blockIdx.y;
  int warp_id = threadIdx.x / 32;

  __shared__ half __align__(16) block_a[2][16][136]; 
  __shared__ half __align__(16) block_b[2][16][72];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int step = 0; step < 4; ++step) {
    int logical_id = step * 128 + threadIdx.x;
    int r = logical_id / 32;
    int c = (logical_id % 32) * 4;
    float4 vec_a = reinterpret_cast<float4*>(&d_a[r * dim_m + offset_a_m + c])[0];
    block_a[0][r][c + 0] = __float2half(vec_a.x);
    block_a[0][r][c + 1] = __float2half(vec_a.y);
    block_a[0][r][c + 2] = __float2half(vec_a.z);
    block_a[0][r][c + 3] = __float2half(vec_a.w);
  }
  for (int step = 0; step < 2; ++step) {
    int logical_id = step * 128 + threadIdx.x;
    int n_idx = logical_id / 4;
    int k_idx = (logical_id % 4) * 4;
    float4 vec_b = reinterpret_cast<float4*>(&d_b[(offset_b_n + n_idx) * dim_k + k_idx])[0];
    block_b[0][k_idx + 0][n_idx] = __float2half(vec_b.x);
    block_b[0][k_idx + 1][n_idx] = __float2half(vec_b.y);
    block_b[0][k_idx + 2][n_idx] = __float2half(vec_b.z);
    block_b[0][k_idx + 3][n_idx] = __float2half(vec_b.w);
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

  int write_idx = 0;
  
  for (int k = 16; k < dim_k; k += 16) {
    write_idx = 1 - write_idx;
    int read_idx = 1 - write_idx;

    for (int step = 0; step < 4; ++step) {
      int logical_id = step * 128 + threadIdx.x;
      int r = logical_id / 32;
      int c = (logical_id % 32) * 4;
      float4 vec_a = reinterpret_cast<float4*>(&d_a[(k + r) * dim_m + offset_a_m + c])[0];
      block_a[write_idx][r][c + 0] = __float2half(vec_a.x);
      block_a[write_idx][r][c + 1] = __float2half(vec_a.y);
      block_a[write_idx][r][c + 2] = __float2half(vec_a.z);
      block_a[write_idx][r][c + 3] = __float2half(vec_a.w);
    }
    for (int step = 0; step < 2; ++step) {
      int logical_id = step * 128 + threadIdx.x;
      int n_idx = logical_id / 4;
      int k_idx = (logical_id % 4) * 4;
      float4 vec_b = reinterpret_cast<float4*>(&d_b[(offset_b_n + n_idx) * dim_k + k + k_idx])[0];
      block_b[write_idx][k_idx + 0][n_idx] = __float2half(vec_b.x);
      block_b[write_idx][k_idx + 1][n_idx] = __float2half(vec_b.y);
      block_b[write_idx][k_idx + 2][n_idx] = __float2half(vec_b.z);
      block_b[write_idx][k_idx + 3][n_idx] = __float2half(vec_b.w);
    }

    for (int r = 0; r < 2; r++) {
      int row_tile = warp_id * 2 + r;
      wmma::load_matrix_sync(a_frag[r], &block_a[read_idx][0][row_tile * 16], 136);
    }
    for (int c = 0; c < 4; c++) {
      wmma::load_matrix_sync(b_frag[c], &block_b[read_idx][0][c * 16], 72);
    }
    for (int r = 0; r < 2; r++) {
      for (int c = 0; c < 4; c++) {
        wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
      }
    }
    __syncthreads();
  }

  int read_idx = write_idx;
  for (int r = 0; r < 2; r++) {
    int row_tile = warp_id * 2 + r;
    wmma::load_matrix_sync(a_frag[r], &block_a[read_idx][0][row_tile * 16], 136);
  }
  for (int c = 0; c < 4; c++) {
    wmma::load_matrix_sync(b_frag[c], &block_b[read_idx][0][c * 16], 72);
  }
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 4; c++) {
      wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
    }
  }

  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 4; c++) {
      int c_m = offset_a_m + (warp_id * 2 + r) * 16;
      int c_n = offset_b_n + c * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;                                          //rows of A and rows C
  int k = 4096;                                           //collums of A and rows of B
  int n = 8192;                                           //collums of B and collums of C
  float alpha = 1.0;                                      //For generalizes matrix multiplicaton
  float beta = 0.0;
  int Nt = 10;                                            //iterations for the kernal to average results over some runs
  float *A, *B, *C, *C2;                                  //C2 as a reference to look for errors; All stored as pointers
  cudaMallocManaged(&A, m * k * sizeof(float));           //create unified memory that can be accessed for GPU and CPU
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)                                 //filling all the matracies; initialize C to 0
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);                           //internal cublas stuff to deal with hardware recourses
  auto tic = chrono::steady_clock::now();                 //Probably not necessary
  for (int i = 0; i < Nt+2; i++) {                        //warm up
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,                           //as defined above
		 CUBLAS_OP_N,                                         //no transpose
		 CUBLAS_OP_N,
		 m,                                                   //sizes of matracies
		 n,
		 k,
		 &alpha,                                              //requires pointers because of fortran background
		 A, CUDA_R_32F, m,                                    //pointer, data-type & lengh of rows
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

  //own implementation starts here
  int tile_m = 128;
  int tile_n = 64;                
  //dim3 helps with threadIdx.x and y (optional z) later          
  dim3 block = dim3(tile_m);                                                //comment might be outdated: amount of threads started in GPU = 64; do not use all to: don't overfloat L1 cache, apparently some kind of sweet spot
  dim3 grid = dim3((m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n);   //amount of blocks started in GPU, dim 0 and 1 get multiplied
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();      //warmup again
    kernel<<< grid, block >>>(m,                        //this leads to exactly 64 entries for every thread  
			      n,
			      k,
			      A,
			      B,
			      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);               //calculate error
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
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
 */

 /*
 * 9. K-Dimension Scaling (K=32): 
 * Increased K-Tile size from 16 to 32.
 * Doubled arithmetic intensity by feeding 32 elements to Tensor Cores per loop.
 * Retained 128 threads and optimal register count.
 * Performance:  58874.55 Gflops
 */
