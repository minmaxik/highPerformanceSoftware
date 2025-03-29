#include <omp.h>
#include <cstdio>

int main() {
#pragma omp parallel
  {
    printf("Hello from thread #%d\n", omp_get_thread_num());
  }
  return 0;
}
