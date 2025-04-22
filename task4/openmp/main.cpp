#include <chrono>
#include <cstdio>
#include <omp.h>
#include <random>

#define TESTS_COUNT 100

static std::random_device rd;
static std::mt19937_64 rng{rd()};

double gen() {
  static std::uniform_real_distribution<double> urd(-100., 100.);
  return urd(rng);
}

int main() {
  const int sizes[] = {1, 10, 100, 1'000, 3'000};
  for (const int n : sizes) {
    std::chrono::duration<double, std::milli> total_time{0};

    for (int _ = 0; _ < TESTS_COUNT; ++_) {
      double *A, *B, *C;

      const int size = n * n;

      A = new double[size];
      B = new double[size];
      C = new double[size];

#pragma omp parallel for
      for (int i = 0; i < size; ++i) {
        A[i] = gen();
        B[i] = gen();
      }

      auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2)
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          double sum = .0;
          for (int k = 0; k < n; ++k) {
            sum += A[i * n + k] * B[k * n + j];
          }
          C[i * n + j] = sum;
        }
      }

      auto end = std::chrono::high_resolution_clock::now();
      total_time += end - start;

      delete[] A;
      delete[] B;
      delete[] C;
    }
    printf("size: %d, avg_time: %f\n", n, total_time.count() / TESTS_COUNT);
  }
  return 0;
}
