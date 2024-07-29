
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <windows.h>
#include <math.h>

#define PI 3.14159265358979323846

__global__
void function(size_t N, double* xr, double* xi, double* x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    double theta;

    for (int k = index; k < N; k += stride) {
        xr[k] = 0;
        xi[k] = 0;
        for (int n = 0; n < N; n++) {
            theta = (2 * PI * k * n) / N;
            xr[k] = xr[k] + x[n] * cos(theta);
            xi[k] = xi[k] - x[n] * sin(theta);
        }
    }

}
__global__
void function2(size_t N, double* xr, double* xi, double* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double theta;
    for (int n = index; n < N; n += stride) {
        y[n] = 0;
        for (int k = 0; k < N; k++) {
            theta = (2 * PI * k * n) / (double)N;
            y[n] = y[n] + xr[k] * cos(theta) - xi[k] * sin(theta);
        }
        y[n] = y[n] / (double)N;

    }

}

int main() {
    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
    LARGE_INTEGER Frequency;
    QueryPerformanceFrequency(&Frequency);
    double total_time, ave_time;
    //const size_t ARRAY_SIZE = 1<<10;
    //const size_t ARRAY_SIZE = 1<<12;
    //const size_t ARRAY_SIZE = 1<<15;
    //const size_t ARRAY_SIZE = 1<<16;
    //const size_t ARRAY_SIZE = 1<<17;
    const size_t ARRAY_SIZE = 1 << 20;
    const size_t ARRAY_BYTES = ARRAY_SIZE * sizeof(double);
    //number of times the program is to be executed
    const size_t loope = 1;
    //declare array

    int device = -1;
    cudaGetDevice(&device);
    double* xr, * xi, * x, * y;
    cudaMallocManaged(&xr, ARRAY_BYTES);
    cudaMallocManaged(&xi, ARRAY_BYTES);
    cudaMallocManaged(&x, ARRAY_BYTES);
    cudaMallocManaged(&y, ARRAY_BYTES);
    //mem advise
    cudaMemAdvise(x, ARRAY_BYTES, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(x, ARRAY_BYTES, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
    //page creation
    cudaMemPrefetchAsync(x, ARRAY_BYTES, cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(xr, ARRAY_BYTES, device, NULL);
    cudaMemPrefetchAsync(xi, ARRAY_BYTES, device, NULL);
    cudaMemPrefetchAsync(y, ARRAY_BYTES, device, NULL);
    // init array
    for (int i = 0;i < ARRAY_SIZE;i++) {
        x[i] = (double)i;
    }

    /*for (size_t i = 0; i < ARRAY_SIZE;i++) {
        printf("y[%d] = %.2f\n", i, x[i]);
    }*/
    //prefetch
    cudaMemPrefetchAsync(x, ARRAY_BYTES, device, NULL);

    // setup CUDA kernel
    //size_t numThreads = 256;
    //size_t numThreads = 512;
    size_t numThreads = 1024;
    //size_t numBlocks = 1;
    size_t numBlocks = (ARRAY_SIZE + numThreads - 1) / numThreads;
    printf("*** function ***\n");
    printf("numElements = %lu\n", ARRAY_SIZE);
    printf("numBlocks = %lu, numThreads = %lu \n", numBlocks, numThreads);
    QueryPerformanceCounter(&StartingTime);
    for (size_t i = 0; i < loope;i++) {
        function << <numBlocks, numThreads >> > (ARRAY_SIZE, xr, xi, x);
    }

    //barrier
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&EndingTime);
    total_time = ((double)((EndingTime.QuadPart - StartingTime.QuadPart) * 1000000 / Frequency.QuadPart)) / 1000;
    ave_time = total_time / loope;
    printf("Time taken for DFT: %f ms\n\n", ave_time);
    cudaMemPrefetchAsync(x, ARRAY_BYTES, cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(xr, ARRAY_BYTES, cudaCpuDeviceId, NULL);
    cudaMemPrefetchAsync(xi, ARRAY_BYTES, cudaCpuDeviceId, NULL);
    //error checking
    /*for (size_t i = 0; i < ARRAY_SIZE;i++) {
        printf("%.3f + j(%.5f)\n", xr[i], xi[i]);
    }*/


    //mem advise
    cudaMemAdvise(xr, ARRAY_BYTES, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(xr, ARRAY_BYTES, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
    cudaMemAdvise(xi, ARRAY_BYTES, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(xi, ARRAY_BYTES, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
    //page creation
    

    cudaMemPrefetchAsync(xr, ARRAY_BYTES, device, NULL);
    cudaMemPrefetchAsync(xi, ARRAY_BYTES, device, NULL);

    QueryPerformanceCounter(&StartingTime);
    function2 << <numBlocks, numThreads >> > (ARRAY_SIZE, xr, xi, y);

    ////barrier
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&EndingTime);
    total_time = ((double)((EndingTime.QuadPart - StartingTime.QuadPart) * 1000000 / Frequency.QuadPart)) / 1000;
    ave_time = total_time / loope;
    printf("Time taken for IDFT: %f ms\n\n", ave_time);
    cudaMemPrefetchAsync(y, ARRAY_BYTES, cudaCpuDeviceId, NULL);
   /* for (size_t i = 0; i < ARRAY_SIZE;i++) {
        printf("y[%d] = %.2f\n", i, y[i]);
    }*/
    size_t err_count = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (x[i]*0.999 > y[i] || x[i] * 1.001 < y[i]) {

            printf("x[%d] = %.2f\ny[%d] = %.2f\n", i, x[i], i, y[i]);
            err_count++;
        }
    }
    printf("Error count(CUDA program): %zu\n", err_count);
    //free memory
    cudaFree(xr);
    cudaFree(xi);
    cudaFree(x);
    cudaFree(y);
    return 0;
}
