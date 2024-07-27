
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

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
    const size_t ARRAY_SIZE = 5;//1<<8;
    //const size_t ARRAY_SIZE = 1<<10;
    //const size_t ARRAY_SIZE = 1<<24;
    //const size_t ARRAY_SIZE = 1<<26;
    //const size_t ARRAY_SIZE = 1<<28;
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

    for (size_t i = 0; i < ARRAY_SIZE;i++) {
        printf("y[%d] = %.2f\n", i, x[i]);
    }
    //prefetch
    cudaMemPrefetchAsync(x, ARRAY_BYTES, device, NULL);

    // setup CUDA kernel
    size_t numThreads = 256;
    //size_t numThreads = 512;
    //size_t numThreads = 1024;
    //size_t numBlocks = 1;
    size_t numBlocks = (ARRAY_SIZE + numThreads - 1) / numThreads;
    printf("*** function ***\n");
    printf("numElements = %lu\n", ARRAY_SIZE);
    printf("numBlocks = %lu, numThreads = %lu \n", numBlocks, numThreads);
    for (size_t i = 0; i < loope;i++) {
        function << <numBlocks, numThreads >> > (ARRAY_SIZE, xr, xi, x);
    }

    //barrier
    cudaDeviceSynchronize();
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
    for (size_t i = 0; i < loope;i++) {
        function2 << <numBlocks, numThreads >> > (ARRAY_SIZE, xr, xi, y);
    }

    ////barrier
    cudaDeviceSynchronize();
    cudaMemPrefetchAsync(y, ARRAY_BYTES, cudaCpuDeviceId, NULL);
    /*for (size_t i = 0; i < ARRAY_SIZE;i++) {
        printf("y[%d] = %.2f\n", i, x[i]);
    }*/
    size_t err_count = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (x[i] != y[i]) {
            err_count++;
        }
    }
    printf("Error count(CUDA program): %zu\n", err_count);
    //free memory
    cudaFree(xr);
    cudaFree(xi);
    cudaFree(x);
    return 0;
}
/*#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}*/
