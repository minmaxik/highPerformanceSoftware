#include <chrono>
#include <cstdio>
#include <omp.h>
#include <random>
#include <ratio>

static std::random_device rd;
static std::mt19937_64 rng{rd()};

double gen() {
  static std::uniform_real_distribution<double> urd(-100., 100.);
  return urd(rng);
}

const int TESTS_COUNT = 100;

int main() {
  const int sizes[] = {1, 10, 1'000, 100'000, 10'000'000};
  for (const int size : sizes) {

    std::chrono::duration<double, std::milli> total_time;

    for (int _ = 0; _ < TESTS_COUNT; ++_) {
      double *data = new double[size];
      for (int i = 0; i < size; ++i) {
        data[i] = gen();
      }
      auto start = std::chrono::high_resolution_clock::now();
      double sum = .0;
#pragma omp parallel for reduction(+ : sum)
      for (int i = 0; i < size; ++i) {
        sum += data[i];
      }

      auto end = std::chrono::high_resolution_clock::now();

      total_time += end - start;
      delete[] data;
    }

    printf("size: %d, avg_time: %f\n", size, total_time.count() / TESTS_COUNT);
  }
  return 0;
}
