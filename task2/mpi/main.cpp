#include <chrono>
#include <cstdio>
#include <mpi.h>
#include <random>
#include <ratio>

static std::random_device rd;
static std::mt19937_64 rng{rd()};

double gen() {
  static std::uniform_real_distribution<double> urd(-100., 100.);
  return urd(rng);
}

const int TESTS_COUNT = 100;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int sizes[] = {1, 10, 1'000, 100'000, 10'000'000};
  for (const int n : sizes) {
    std::chrono::duration<double, std::milli> total_time;

    for (int _ = 0; _ < TESTS_COUNT; ++_) {
      double *data = nullptr;
      if (rank == 0) {
        data = new double[n];
        for (int i = 0; i < n; ++i) {
          data[i] = gen();
        }
      }

      int *counts = new int[size];
      int *displs = new int[size];

      int base = n / size;
      int remainder = n % size;
      for (int i = 0; i < size; ++i) {
        counts[i] = base + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
      }

      int local_n = counts[rank];
      double *local_data = new double[local_n];

      double local_sum = .0;
      double global_sum = .0;

      auto start = std::chrono::high_resolution_clock::now();

      MPI_Scatterv(data, counts, displs, MPI_DOUBLE, local_data, local_n,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);

      for (int i = 0; i < local_n; ++i) {
        local_sum += local_data[i];
      }

      MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
                 MPI_COMM_WORLD);

      auto end = std::chrono::high_resolution_clock::now();

      if (rank == 0) {
        total_time += end - start;
        delete[] data;
      }
      delete[] counts;
      delete[] displs;
      delete[] local_data;
    }
    if (rank == 0) {
      printf("size: %d, avg_time: %f\n", n, total_time.count() / TESTS_COUNT);
    }
  }
  MPI_Finalize();
  return 0;
}
