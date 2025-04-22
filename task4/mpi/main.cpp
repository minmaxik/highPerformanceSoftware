#include <chrono>
#include <cstdio>
#include <mpi.h>
#include <random>

#define TESTS_COUNT 100

static std::random_device rd;
static std::mt19937_64 rng{rd()};

double gen() {
  static std::uniform_real_distribution<double> urd(-100., 100.);
  return urd(rng);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  const int sizes[] = {1, 10, 100, 1'000, 3'000};
  for (const int n : sizes) {
    std::chrono::duration<double, std::milli> total_time{0};

    const int n_size = n * n;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int _ = 0; _ < TESTS_COUNT; ++_) {
      double *A = nullptr, *B = nullptr, *C = nullptr;
      B = new double[n_size];
      if (rank == 0) {
        A = new double[n_size];
        C = new double[n_size];

        for (int i = 0; i < n_size; ++i) {
          A[i] = gen();
          B[i] = gen();
        }
      }

      int *counts = new int[size];
      int *displs = new int[size];

      int base = n / size;
      int remainder = n % size;
      for (int i = 0; i < size; ++i) {
        counts[i] = (base + (i < remainder ? 1 : 0)) * n;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
      }

      int local_n = counts[rank];
      int local_rows = counts[rank] / n;

      double *local_A = new double[local_n];
      double *local_C = new double[local_n];

      auto start = std::chrono::high_resolution_clock::now();

      MPI_Scatterv(A, counts, displs, MPI_DOUBLE, local_A, local_n, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

      MPI_Bcast(B, n_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      for (int row = 0; row < local_rows; ++row) {
        for (int col = 0; col < n; ++col) {
          double sum = .0;
          for (int k = 0; k < n; ++k) {
            sum += local_A[row * n + k] * B[k * n + col];
          }
          local_C[row * n + col] = sum;
        }
      }

      MPI_Gatherv(local_C, local_n, MPI_DOUBLE, C, counts, displs, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);

      auto end = std::chrono::high_resolution_clock::now();

      if (rank == 0) {
        total_time += end - start;

        delete[] A;
        delete[] C;
      }

      delete[] B;
      delete[] counts;
      delete[] displs;
      delete[] local_A;
      delete[] local_C;
    }
    if (rank == 0) {
      printf("size: %d, avg_time: %f\n", n, total_time.count() / TESTS_COUNT);
    }
  }

  MPI_Finalize();
  return 0;
}
