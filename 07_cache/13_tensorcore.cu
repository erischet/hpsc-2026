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
  int offset_a_m = 128 * blockIdx.x;                           //for e.g. block 2 --> 64
  int offset_b_n = 64 * blockIdx.y;
  //int i = threadIdx.x;
  int warp_id = threadIdx.x / 32;                             //32 is a hardware value of the gpu

  // +8 verschiebt die Speicheradressen und löst die Bank Conflicts auf --> Imporvements by almost 100 %
  __shared__ half __align__(16) block_a[16][128 + 8]; 
  __shared__ half __align__(16) block_b[16][64 + 8];                          //probably introduces error compared to other programms

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4]; //reserving storage in register for tensor cores;
                                  //m,  n,  k,  
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);                   //set values to 0 for initialisation

  for (int k = 0; k < dim_k; k += 16) {
      __syncthreads(); 
      
      for (int step = 0; step < 4; ++step) {
        int logical_id = step * 128 + threadIdx.x; // ID von 0 bis 511
        int r = logical_id / 32;                   // Zeile (0 bis 15)
        int c = (logical_id % 32) * 4;             // Spalte (0, 4, 8 ... 124)
        
        float4 vec_a = reinterpret_cast<float4*>(&d_a[(k + r) * dim_m + offset_a_m + c])[0];
        block_a[r][c + 0] = __float2half(vec_a.x);
        block_a[r][c + 1] = __float2half(vec_a.y);
        block_a[r][c + 2] = __float2half(vec_a.z);
        block_a[r][c + 3] = __float2half(vec_a.w);
      }

      // 2. Matrix B laden (2 Schritte pro Thread, koalesziert)
      for (int step = 0; step < 2; ++step) {
        int logical_id = step * 128 + threadIdx.x; // ID von 0 bis 255
        int n_idx = logical_id / 4;                // Spalte (0 bis 63)
        int k_idx = (logical_id % 4) * 4;          // Zeile (0, 4, 8, 12)
        
        float4 vec_b = reinterpret_cast<float4*>(&d_b[(offset_b_n + n_idx) * dim_k + k + k_idx])[0];
        block_b[k_idx + 0][n_idx] = __float2half(vec_b.x);
        block_b[k_idx + 1][n_idx] = __float2half(vec_b.y);
        block_b[k_idx + 2][n_idx] = __float2half(vec_b.z);
        block_b[k_idx + 3][n_idx] = __float2half(vec_b.w);
      }
      __syncthreads(); 
      //loading the data is finished

    //Improvements for 08 - not sure if really of advantage, performance almost doesn't get better
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

    for (int r = 0; r < 2; r++) {
      int row_tile = warp_id * 2 + r;
      // Stride 136 (128 nutzbare Daten + 8 Padding)
      wmma::load_matrix_sync(a_frag[r], &block_a[0][row_tile * 16], 136);
    }

    for (int c = 0; c < 4; c++) {
      // Stride 72 (64 nutzbare Daten + 8 Padding)
      wmma::load_matrix_sync(b_frag[c], &block_b[0][c * 16], 72);
    }

    for (int r = 0; r < 2; r++) {
      for (int c = 0; c < 4; c++) {
        wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
      }
    }
  }
  //end improvement 08

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
