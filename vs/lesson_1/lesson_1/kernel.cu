
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


// __device__ - GPU
// __global__ - GPU 
// __host__   - CPU
__global__ void add( int a, int b, int *c)
{
  *c = a + b;
}

int main()
{
  int c;
  int *dev_c;

  cudaMalloc( (void**)&dev_c, sizeof(int));

  add<<<1,1>>> (1, 2, dev_c);

  cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

  printf("%i\n", c);

  cudaFree(dev_c);
  return 0;
}

