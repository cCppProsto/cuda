/*
 ============================================================================
 Name        : lesson_5.cu
 Author      : cppProsto
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include "../../../../common/book.h"
#include <cuda_runtime.h>
#include "../../../../common/helper_cuda.h"
#include "../../../../common/helper_string.h"

#include <sys/time.h>
#include<unistd.h>

#define N   60000

void print_device_info()
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess)
  {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
  {
    printf("There are no available device(s) that support CUDA\n");
  }
  else
  {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev)
  {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
        driverVersion / 1000, (driverVersion % 100) / 10,
        runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
        deviceProp.major, deviceProp.minor);

    char msg[256];
    snprintf(msg, sizeof(msg),
    "  Total amount of global memory:                 %.0f MBytes "
    "(%llu bytes)\n",
    static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
    (unsigned long long)deviceProp.totalGlobalMem);

    printf("%s", msg);

    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
    deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
    deviceProp.multiProcessorCount);
    printf(
    "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
    "GHz)\n",
    deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    #if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    printf("  Memory Clock rate:                             %.0f Mhz\n",
    deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
    deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize)
    {
      printf("  L2 Cache Size:                                 %d bytes\n",
          deviceProp.l2CacheSize);
    }

    #else
    // This only available in CUDA 4.0-4.2 (but these were only exposed in the
    // CUDA Driver API)
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
              dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
    memoryClock * 1e-3f);
    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,
              CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
    memBusWidth);
    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    if (L2CacheSize) {
    printf("  L2 Cache Size:                                 %d bytes\n",
    L2CacheSize);
    }
    #endif

    printf("  Total number of registers available per block: %d\n",
    deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
    deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
    deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
    deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
    deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
    deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
    deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
    deviceProp.maxGridSize[2]);
    printf("  Integrated GPU sharing Host Memory:            %s\n",
    deviceProp.integrated ? "Yes" : "No");
  }
}

namespace gpu_add
{
  __global__ void add( int *a, int *b, int *c )
  {
    int tid = blockIdx.x;    // this thread handles the data at its thread id
    if (tid < N)
      c[tid] = a[tid] + b[tid];
  }

  void gpu_main( void )
  {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++)
    {
      a[i] = -i;
      b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    cudaEvent_t start, stop;
    float gpuTime = 0.0;

    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start, 0 );

    add<<<N,1>>>( dev_a, dev_b, dev_c );

    cudaEventRecord( stop, 0 );

    cudaEventSynchronize( stop );

    cudaEventElapsedTime( &gpuTime, start, stop );
    printf("time on GPU = %.2f miliseconds\n", gpuTime);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );


    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );
  }
}

namespace cpu_add
{

  float timedifference_msec(struct timeval t0, struct timeval t1)
  {
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
  }

  void add( int *a, int *b, int *c )
  {
    int tid = 0;    // this is CPU zero, so we start at zero
    while (tid < N)
    {
      c[tid] = a[tid] + b[tid];
      tid += 1;   // we have one CPU, so we increment by one
    }
  }

  void cpu_main( void )
  {
    int a[N], b[N], c[N];

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++)
    {
      a[i] = -i;
      b[i] = i * i;
    }

    struct timeval stop, start;
    gettimeofday(&start, NULL);

    add( a, b, c );

    gettimeofday(&stop, NULL);
    printf("time on CPU = %.2f miliseconds\n",  timedifference_msec(start, stop));
  }
}

int main()
{
  print_device_info();

  gpu_add::gpu_main();
  cpu_add::cpu_main();
  return 0;
}
