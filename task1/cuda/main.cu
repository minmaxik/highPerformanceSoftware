#include <cstdio>

__global__ void hello() {
  printf("Hello from thread # %d (block %d)\n", threadIdx.x, blockIdx.x);
}

int main() {
  hello<<<2,4>>>();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel failed: %s", cudaGetErrorString(err));
  }
  
  cudaDeviceSynchronize();
  
  return 0;
}
