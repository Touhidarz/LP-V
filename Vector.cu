#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h> // For CUDA functions

using namespace std;
using namespace std::chrono;

#define N (1 << 20)   // Vector size (~1 million)
#define SIZE 512      // Matrix size 512x512

// ---------- Compare two arrays ----------
bool compareArrays(float *arr1, float *arr2, int size, float epsilon = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (fabs(arr1[i] - arr2[i]) > epsilon) {
            cout << "Mismatch at index " << i << ": " << arr1[i] << " vs " << arr2[i] << endl;
            return false;
        }
    }
    return true;
}

// ---------- Vector Addition ----------
__global__ void vectorAddGPU(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void vectorAddCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
}

// ---------- Matrix Multiplication ----------
__global__ void matMulGPU(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0;
        for (int k = 0; k < width; ++k)
            sum += A[row * width + k] * B[k * width + col];
        C[row * width + col] = sum;
    }
}

void matMulCPU(float *A, float *B, float *C, int width) {
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < width; ++j) {
            float sum = 0;
            for (int k = 0; k < width; ++k)
                sum += A[i * width + k] * B[k * width + j];
            C[i * width + j] = sum;
        }
}

int main() {
    // ---------------- VECTOR ADDITION ----------------
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C_cpu = new float[N];
    float *h_C_gpu = new float[N];
    float *d_A, *d_B, *d_C;

    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    auto start_cpu_vec = high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    auto end_cpu_vec = high_resolution_clock::now();
    auto duration_cpu_vec = duration_cast<milliseconds>(end_cpu_vec - start_cpu_vec);
    cout << "\nVector Addition CPU Time: " << duration_cpu_vec.count() << " ms" << endl;

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    auto start_gpu_vec = high_resolution_clock::now();
    vectorAddGPU<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu_vec = high_resolution_clock::now();
    auto duration_gpu_vec = duration_cast<milliseconds>(end_gpu_vec - start_gpu_vec);
    cout << "Vector Addition GPU Time: " << duration_gpu_vec.count() << " ms" << endl;

    cudaMemcpy(h_C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Vector Speedup Factor: " << (float)duration_cpu_vec.count() / duration_gpu_vec.count() << "x" << endl;
    cout << "Vector addition match: " << (compareArrays(h_C_cpu, h_C_gpu, N) ? "Yes ✅" : "No ❌") << endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C_cpu; delete[] h_C_gpu;

    // ---------------- MATRIX MULTIPLICATION ----------------
    int size = SIZE * SIZE;
    float *A = new float[size];
    float *B = new float[size];
    float *C_cpu = new float[size];
    float *C_gpu = new float[size];
    float *d_MatA, *d_MatB, *d_MatC;

    for (int i = 0; i < size; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    cudaMalloc(&d_MatA, size * sizeof(float));
    cudaMalloc(&d_MatB, size * sizeof(float));
    cudaMalloc(&d_MatC, size * sizeof(float));

    cudaMemcpy(d_MatA, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, B, size * sizeof(float), cudaMemcpyHostToDevice);

    auto start_cpu_mat = high_resolution_clock::now();
    matMulCPU(A, B, C_cpu, SIZE);
    auto end_cpu_mat = high_resolution_clock::now();
    auto duration_cpu_mat = duration_cast<milliseconds>(end_cpu_mat - start_cpu_mat);
    cout << "\nMatrix Multiplication CPU Time: " << duration_cpu_mat.count() << " ms" << endl;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((SIZE + 15) / 16, (SIZE + 15) / 16);

    auto start_gpu_mat = high_resolution_clock::now();
    matMulGPU<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_MatB, d_MatC, SIZE);
    cudaDeviceSynchronize();
    auto end_gpu_mat = high_resolution_clock::now();
    auto duration_gpu_mat = duration_cast<milliseconds>(end_gpu_mat - start_gpu_mat);
    cout << "Matrix Multiplication GPU Time: " << duration_gpu_mat.count() << " ms" << endl;

    cudaMemcpy(C_gpu, d_MatC, size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Matrix Speedup Factor: " << (float)duration_cpu_mat.count() / duration_gpu_mat.count() << "x" << endl;
    cout << "Matrix multiplication match: " << (compareArrays(C_cpu, C_gpu, size) ? "Yes ✅" : "No ❌") << endl;

    cudaFree(d_MatA); cudaFree(d_MatB); cudaFree(d_MatC);
    delete[] A; delete[] B; delete[] C_cpu; delete[] C_gpu;

    return 0;
}
