#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// if N >= 65 535 then error
#define N 10000

void add_cpu(int *a, int *b, int *c)
{
  int tid = 0;
  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    tid += 1;
  }
}

/*
// CPU core_1
void add_cpu(int *a, int *b, int *c)
{
  int tid = 0;
  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    tid += 2;
  }
}

// CPU core_2
void add_cpu(int *a, int *b, int *c)
{
  int tid = 1;
  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    tid += 2;
  }
}
*/

__global__ void add_gpu(int *a, int *b, int *c)
{
  int tid = blockIdx.x;
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

/*
// gpu_1 core
__global__ void add_gpu(int *a, int *b, int *c)
{
  int tid = 0;
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

// gpu_2 core
__global__ void add_gpu(int *a, int *b, int *c)
{
  int tid = 1
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

// gpu_3 core
__global__ void add_gpu(int *a, int *b, int *c)
{
  int tid = 2;
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

// gpu_4 core
__global__ void add_gpu(int *a, int *b, int *c)
{
  int tid = 3;
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

*/



int main()
{
  int a[N];
  int b[N];
  int c[N];

  for (int i = 0; i < N; i++)
  {
    a[i] = -i;
    b[i] = i * i;
  }

  add_cpu(a, b, c);

  /*
  for (int i = 0; i < N; i++)
  {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }
  */



  int *dev_a;
  int *dev_b;
  int *dev_c;

  cudaMalloc((void**)&dev_a, N * sizeof(int));
  cudaMalloc((void**)&dev_b, N * sizeof(int));
  cudaMalloc((void**)&dev_c, N * sizeof(int));


  for (int i = 0; i < N; i++)
  {
    a[i] = -i;
    b[i] = i * i;
  }

  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  add_gpu <<<N, 1 >>> (dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  /*
  for (int i = 0; i < N; i++)
  {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }
  */

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 1;
}



/*
for (int i = 0; i < N; i++)
{
  a[i] = -i;
  b[i] = i * i;
}


int *dev_a;
int *dev_b;
int *dev_c;

cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

add<<<N, 1>>>(dev_a, dev_b, dev_c);

cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

for (int i = 0; i < N; i++)
{
  printf("%d + %d = %d\n", a[i], b[i], c[i]);
}

cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);
*/
