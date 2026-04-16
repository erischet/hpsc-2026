/*
Oringinal output:
41 67 134 100 169 124 78 158 162 64 105 145 81 27 161 91 195 142 27 36 
27 27 36 41 64 67 78 81 91 100 105 124 134 142 145 158 161 162 169 195 
*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>


void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) { 
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++]; 
  }
  for (int i=0; i<tmp.size(); i++) 
    vec[begin++] = tmp[i];
}


void merge_sort(std::vector<int>& vec, int begin, int end) {
  if(begin < end) { 
    int mid = (begin + end) / 2;

    #pragma omp task shared(vec)
    merge_sort(vec, begin, mid);

    #pragma omp task shared(vec)
    merge_sort(vec, mid+1, end);

    #pragma omp taskwait
    merge(vec, begin, mid, end);
  }
}


int main() {

 omp_set_num_threads(4);

  int n = 20;
  std::vector<int> vec(n);
  #pragma omp parallel for
  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
  }

  for (int i=0; i<n; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n");

  #pragma omp parallel
  {
  #pragma omp single
  merge_sort(vec, 0, n-1);
  }
  //Ausgabe, kann so bleiben
  for (int i=0; i<n; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n");
}
