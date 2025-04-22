#include <chrono>
#include <cstdio>
#include <random>
#include <ratio>

#define TESTS_COUNT 100
#define BLOCK_SIZE 64

static std::random_device rd;
static std::mt19937_64 rng{rd()};

double gen() {
  static std::uniform_real_distribution<double> urd(-100., 100.);
  return urd(rng);
}

__global__ void matrixMultKernel(const double *A, const double *B, double *C,
                                 int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    double sum = .0;
    for (int i = 0; i < n; ++i) {
      sum += A[row * n + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
  }
}

void matrix_multiply(int n) {
  double *h_A, *h_B, *h_C;
  double *d_A, *d_B, *d_C;
  const size_t size = n * n;

  h_A = new double[size];
  h_B = new double[size];
  h_C = new double[size];

  for (int i = 0; i < size; ++i) {
    h_A[i] = gen();
    h_B[i] = gen();
  }

  cudaMalloc(&d_A, size * sizeof(double));
  cudaMalloc(&d_B, size * sizeof(double));
  cudaMalloc(&d_C, size * sizeof(double));

  cudaMemcpy(d_A, h_A, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size * sizeof(double), cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matrixMultKernel<<<grid, block>>>(d_A, d_B, d_C, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
}

int main() {
  const int sizes[] = {1, 10, 100, 1'000, 3'000};
  for (const int n : sizes) {
    std::chrono::duration<double, std::milli> total_time{0};

    for (int _ = 0; _ < TESTS_COUNT; ++_) {
      auto start = std::chrono::high_resolution_clock::now();
      matrix_multiply(n);
      auto end = std::chrono::high_resolution_clock::now();
      total_time += end - start;
    }
    printf("size: %d, avg_time: %f\n", n, total_time.count() / TESTS_COUNT);
  }
  return 0;
}
