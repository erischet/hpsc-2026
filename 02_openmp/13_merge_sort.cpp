/*
Orinal output:
41 67 134 100 169 124 78 158 162 64 105 145 81 27 161 91 195 142 27 36 
27 27 36 41 64 67 78 81 91 100 105 124 134 142 145 158 161 162 169 195 
*/

#include <cstdio>
#include <cstdlib>
#include <vector>

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
    merge_sort(vec, begin, mid);
    merge_sort(vec, mid+1, end);
    merge(vec, begin, mid, end);
  }
}

int main() {
  int n = 20;
  std::vector<int> vec(n);
  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
    printf("%d ",vec[i]);
  }
  printf("\n");
  merge_sort(vec, 0, n-1);
  for (int i=0; i<n; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n");
}
