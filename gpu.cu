#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA ядро, которое выполняет бесконечную нагрузку
__global__ void load_kernel(float *a, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float val = a[idx];
        while (true) {
            val = val * 1.000001f + 0.000001f;
            if (val > 1e10f) val = 0.0f;
        }
    }
}

// Функция для запуска нагрузки
extern "C" void start_gpu_load() {
    const int N = 1024 * 1024; // 1 миллион элементов
    size_t size = N * sizeof(float);

    float *host_array = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        host_array[i] = 1.0f;
    }

    float *d_array;
    cudaError_t err = cudaMalloc((void**)&d_array, size);
    if (err != cudaSuccess) {
        printf("Ошибка выделения памяти на GPU: %s\n", cudaGetErrorString(err));
        free(host_array);
        return;
    }

    err = cudaMemcpy(d_array, host_array, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Ошибка копирования данных: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        free(host_array);
        return;
    }

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    load_kernel<<<grid_size, block_size>>>(d_array, N);

    // Не вызываем синхронизацию — ядро работает бесконечно
}