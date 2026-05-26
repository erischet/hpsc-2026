#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <chrono>

using namespace std;
using namespace nvcuda;

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32
#define PAD 8

__global__ void kernel_optimized(int dim_m, int dim_n, int dim_k, float *d_a, float *d_b, float *d_c) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int warp_id = tx / 32;
    int lane_id = tx % 32;
    int warp_row = warp_id / 4;
    int warp_col = warp_id % 4;

    int offset_a_m = by * TILE_M;
    int offset_b_n = bx * TILE_N;

    extern __shared__ half smem[];
    half* smem_a = smem; 
    half* smem_b = smem + 2 * TILE_M * (TILE_K + PAD);
    float* smem_c = reinterpret_cast<float*>(smem); // Wiederverwendung für Epilog

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            wmma::fill_fragment(acc[r][c], 0.0f);
        }
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

    int load_a_row = tx / 8;
    int load_a_col = (tx % 8) * 4;
    int load_b_row = tx / 32;
    int load_b_col = (tx % 32) * 4;

    for (int k = 0; k < dim_k; k += TILE_K) {
        int smem_idx = (k / TILE_K) % 2;
        int smem_offset_a = smem_idx * TILE_M * (TILE_K + PAD);
        int smem_offset_b = smem_idx * TILE_K * (TILE_N + PAD);

        // Lade A (float4) -> Konvertiere zu half2 -> Speichere in SMEM
        if (offset_a_m + load_a_row < dim_m && k + load_a_col < dim_k) {
            float4 vec_a = reinterpret_cast<float4*>(&d_a[(offset_a_m + load_a_row) * dim_m + k + load_a_col])[0];
            half2 h0 = __float2half2_rn(make_float2(vec_a.x, vec_a.y));
            half2 h1 = __float2half2_rn(make_float2(vec_a.z, vec_a.w));
            smem_a[smem_offset_a + load_a_col * TILE_M + load_a_row] = h0.x;
            smem_a[smem_offset_a + (load_a_col + 1) * TILE_M + load_a_row] = h0.y;
            smem_a[smem_offset_a + (load_a_col + 2) * TILE_M + load_a_row] = h1.x;
            smem_a[smem_offset_a + (load_a_col + 3) * TILE_M + load_a_row] = h1.y;
        }

        // Lade B (float4) -> Konvertiere zu half2 -> Speichere in SMEM
        if (k + load_b_row < dim_k && offset_b_n + load_b_col < dim_n) {
            float4 vec_b = reinterpret_cast<float4*>(&d_b[(offset_b_n + load_b_col) * dim_k + k + load_b_row])[0];
            half2 h0 = __float2half2_rn(make_float2(vec_b.x, vec_b.y));
            half2 h1 = __float2half2_rn(make_float2(vec_b.z, vec_b.w));
            smem_b[smem_offset_b + load_b_row * (TILE_N + PAD) + load_b_col] = h0.x;
            smem_b[smem_offset_b + load_b_row * (TILE_N + PAD) + load_b_col + 1] = h0.y;
            smem_b[smem_offset_b + load_b_row * (TILE_N + PAD) + load_b_col + 2] = h1.x;
            smem_b[smem_offset_b + load_b_row * (TILE_N + PAD) + load_b_col + 3] = h1.y;
        }
        __syncthreads();

        // Berechne MMA für den aktuellen Block
        for (int step = 0; step < TILE_K; step += 16) {
            for (int r = 0; r < 4; r++) {
                wmma::load_matrix_sync(a_frag[r], &smem_a[smem_offset_a + step * TILE_M + (warp_row * 64 + r * 16)], TILE_M);
            }
            for (int c = 0; c < 4; c++) {
                wmma::load_matrix_sync(b_frag[c], &smem_b[smem_offset_b + step * (TILE_N + PAD) + (warp_col * 64 + c * 16)], TILE_N + PAD);
            }
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
                }
            }
        }
        __syncthreads();
    }

    // Epilog: Schreibe Fragmente in Shared Memory
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            wmma::store_matrix_sync(&smem_c[(warp_row * 64 + r * 16) * TILE_N + (warp_col * 64 + c * 16)], acc[r][c], TILE_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Epilog: Koaleszierter Transfer vom Shared Memory in den globalen Speicher
    int store_r = tx / 32;
    int store_c = (tx % 32) * 4;
    for (int i = 0; i < 4; i++) {
        int current_row = store_r + i * 32;
        if (offset_a_m + current_row < dim_m && offset_b_n + store_c < dim_n) {
            float4 out_val;
            out_val.x = smem_c[current_row * TILE_N + store_c + 0];
            out_val.y = smem_c[current_row * TILE_N + store_c + 1];
            out_val.z = smem_c[current_row * TILE_N + store_c + 2];
            out_val.w = smem_c[current_row * TILE_N + store_c + 3];
            reinterpret_cast<float4*>(&d_c[(offset_b_n + store_c) * dim_m + offset_a_m + current_row])[0] = out_val;
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
    
    for (int i=0; i<m; i++) for (int j=0; j<k; j++) A[k*i+j] = drand48();
    for (int i=0; i<k; i++) for (int j=0; j<n; j++) B[n*i+j] = drand48();
    for (int i=0; i<n; i++) for (int j=0; j<m; j++) C[m*i+j] = C2[m*i+j] = 0;
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
            &alpha, A, CUDA_R_32F, m, B, CUDA_R_32F, k,
            &beta, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
    double cublas_flops = double(num_flops) / tcublas / 1.0e9;

    dim3 block(256);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    
    int smem_size_a = 2 * TILE_M * (TILE_K + PAD) * sizeof(half);
    int smem_size_b = 2 * TILE_K * (TILE_N + PAD) * sizeof(half);
    int smem_size_c = TILE_M * TILE_N * sizeof(float);
    int smem_size = max(smem_size_a + smem_size_b, smem_size_c);
    
    cudaFuncSetAttribute(kernel_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_optimized<<<grid, block, smem_size>>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
    double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
    
    printf("CUBLAS: %.2f Gflops, Custom: %.2f Gflops\n", cublas_flops, cutlass_flops);
    
    double err = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            err += fabs(C[m*i+j] - C2[m*i+j]);
        }
    }
    printf("Error: %lf\n", err/n/m);
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C2);
    cublasDestroy(cublas_handle);
    return 0;
}