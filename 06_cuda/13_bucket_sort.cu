#include <cstdio>
#include <iostream>

__global__ void create_buckets(int *bucket) {
    bucket[threadIdx.x] = 0;
}

__global__ void count_keys(int *key, int *bucket, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        atomicAdd(&bucket[key[i]], 1);
    }
}

__global__ void offset_calc(int *bucket, int *offsets, int *b, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= range) return;

    // Shift right for exclusive scan. First element is always 0.
    offsets[i] = (i == 0) ? 0 : bucket[i - 1];
    __syncthreads();

    // Kogge-Stone Prefix Sum
    for(int j = 1; j < range; j <<= 1) {
        b[i] = offsets[i]; // Store current state in temporary buffer b
        __syncthreads();

        if(i >= j) {
            offsets[i] += b[i - j];
        }
        __syncthreads();
    }
}

__global__ void scatter_kernel(int *key, int *bucket, int *offsets, int range) {
    int val = threadIdx.x;

    if (val < range) {
        int start = offsets[val];
        int count = bucket[val];

        for (int i = 0; i < count; i++) {
            key[start + i] = val;
        }
    }
}

int main() {
    int n = 50;
    int range = 5;

    int const M = 1024;

    int *key;
    cudaMallocManaged(&key, n * sizeof(int));

    int *bucket;
    cudaMallocManaged(&bucket, range * sizeof(int));

    int *offsets;
    cudaMallocManaged(&offsets, range * sizeof(int));

    int *b;
    cudaMallocManaged(&b, range*sizeof(int));

    for (int i=0; i<n; i++) {
        key[i] = rand() % range;
        printf("%d ",key[i]);
    }
    printf("\n");

    //Ab hier umschreiben
    create_buckets<<<1,range>>>(bucket);

    count_keys<<<(n+M-1)/M,M>>>(key,bucket,n);

    offset_calc<<<1,range>>>(bucket,offsets,b,range);

    scatter_kernel<<<1,range>>>(key,bucket,offsets,range);

    cudaDeviceSynchronize();

    //for verification if offset_calc works propperly
    //for (int i=0; i<range; i++) {
    //printf("%d ",offsets[i]);
    //}

    /*for (int i=0, j=0; i<range; i++) {
        for (; bucket[i]>0; bucket[i]--) {
            key[j++] = i;
        }
    }
    */

    for (int i=0; i<n; i++) {
        printf("%d ",key[i]);
    }

    std::cout << "\n";

    cudaFree(key);
    cudaFree(bucket);
    cudaFree(b);
    cudaFree(offsets);

    return 0;
}