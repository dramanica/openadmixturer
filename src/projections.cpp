// Port of OpenADMIXTURE.jl src/projections.jl (project_q!, project_p!).
#include "projections.h"
#include <algorithm>
#include <functional>
#include <vector>

// Projection onto {sum(q) == 1, q >= eps}: shift by eps, project onto the
// simplex of mass tau = 1 - K * eps with the sort-and-threshold algorithm,
// then shift back. Julia sorts an index permutation with its own quicksort;
// only the sorted values matter, so we sort a copy of the values instead.
void project_q(double* q, int K, double eps) {
  const int stack_max = 64;
  double stack_buf[stack_max];
  std::vector<double> heap_buf;
  double* b = stack_buf;
  if (K > stack_max) {
    heap_buf.resize(K);
    b = heap_buf.data();
  }
  const double tau = 1.0 - K * eps;
  for (int k = 0; k < K; k++) {
    q[k] -= eps;
    b[k] = q[k];
  }
  std::sort(b, b + K, std::greater<double>());
  // find the threshold: the first i at which the running mean exceeds the
  // next largest value; otherwise use all K entries
  double tsum = 0, tmax = 0;
  bool found = false;
  for (int i = 0; i < K - 1; i++) {
    tsum += b[i];
    tmax = (tsum - tau) / (i + 1);
    if (tmax >= b[i + 1]) {
      found = true;
      break;
    }
  }
  if (!found) tmax = (tsum + b[K - 1] - tau) / K;
  for (int k = 0; k < K; k++) q[k] = std::max(q[k] - tmax, 0.0) + eps;
}

void project_p(double* p, int K, double eps) {
  for (int k = 0; k < K; k++) p[k] = std::min(std::max(p[k], eps), 1.0 - eps);
}
