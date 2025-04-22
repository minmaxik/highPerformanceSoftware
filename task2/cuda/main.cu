#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

static std::random_device rd;
static std::mt19937_64 rng{rd()};

double gen() {
  static std::uniform_real_distribution<double> urd(-100., 100.);
  return urd(rng);
}

#define BLOCK_SIZE 512
const int TESTS_COUNT = 100;

__global__ void sum_kernel(const double *input, double *output, int n) {
  __shared__ double sdata[BLOCK_SIZE];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (i < n) ? input[i] : .0;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    output[blockIdx.x] = sdata[0];
}

double cuda_sum(const double *input, int n) {
  double *cuda_input = nullptr, *cuda_output = nullptr;
  int output_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaMalloc(&cuda_input, n * sizeof(double));
  cudaMalloc(&cuda_output, output_size * sizeof(double));

  cudaMemcpy(cuda_input, input, n * sizeof(double), cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE);
  dim3 grid(output_size);

  sum_kernel<<<grid, block>>>(cuda_input, cuda_output, n);

  std::vector<double> output(output_size);
  cudaMemcpy(output.data(), cuda_output, output_size * sizeof(double),
             cudaMemcpyDeviceToHost);

  double total_sum = .0;
  for (double val : output) {
    total_sum += val;
  }

  cudaFree(cuda_input);
  cudaFree(cuda_output);

  return total_sum;
}

int main() {
  const int sizes[] = {1, 10, 1'000, 100'000, 10'000'000};
  for (const int n : sizes) {
    std::chrono::duration<double, std::milli> total_time{0};

    for (int _ = 0; _ < TESTS_COUNT; ++_) {
      std::vector<double> data(n);

      for (double &val : data) {
        val = gen();
      }

      auto start = std::chrono::high_resolution_clock::now();

      cuda_sum(data.data(), n);

      auto end = std::chrono::high_resolution_clock::now();

      total_time += end - start;
    }
    printf("size: %d, avg_time: %f\n", n, total_time.count() / TESTS_COUNT);
  }
  return 0;
}
