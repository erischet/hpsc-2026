/*
Orinal output:
3.142425985001098
*/

#include <cstdio>
#include <omp.h>

int main() {
  omp_set_num_threads(4);
  int n = 10; //Tested with Measure-Command { ./a.exe }; Takes n > 1000000 to see actual improvements
  double dx = 1. / n;
  double pi = 0;
  #pragma omp parallel for reduction(+:pi)
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }

  printf("%17.15f\n",pi);
}

//3.142425985001098 - good
