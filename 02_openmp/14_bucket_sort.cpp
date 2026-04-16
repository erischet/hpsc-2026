/*
Oringinal output:
1 2 4 0 4 4 3 3 2 4 0 0 1 2 1 1 0 2 2 1 1 4 2 3 2 2 1 1 3 0 2 1 1 3 4 2 2 4 0 4 3 1 2 3 3 4 1 1 3 3 
0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 
*/


#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {

  omp_set_num_threads(4);
  int n = 50;
  int range = 5;
  std::vector<int> key(n);

  #pragma omp parallel for
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
  }
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  // 1. Zählen (Parallel mit atomic)
  std::vector<int> bucket(range, 0); 
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    // atomic verhindert, dass Threads sich beim Hochzählen überschreiben
    #pragma omp atomic
    bucket[key[i]]++;
  }

  // 2. Offsets berechnen (Seriell)
  // Da range (5) sehr klein ist, ist die serielle Prefix Summe hier schneller
  std::vector<int> offset(range, 0);
  for (int i = 1; i < range; i++) {
    offset[i] = offset[i - 1] + bucket[i - 1];
  }

  // 3. Einsortieren (Seriell)
  // Dieser Teil bleibt seriell, da j++ von den vorherigen Werten abhängt
  for (int i = 0; i < range; i++) {
    int j = offset[i];
    for (; bucket[i] > 0; bucket[i]--) {
      key[j++] = i;
    }
  }

  // 4. Finale Ausgabe
  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");

  return 0;
}