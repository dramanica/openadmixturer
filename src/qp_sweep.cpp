// STUB - to be replaced by the real port of qp.jl
#include "qp_sweep.h"
#include "projections.h"
#include <Rcpp.h>

QPWork::QPWork(int K) : K(K), tab((K + 2) * (K + 2)), diag(K + 2),
  swept(K + 2), V(K * K), tmp(K) {}

int qp_solve_column(const double* H, const double* grad, const double* par,
                    int K, bool simplex, double* delta, QPWork& work) {
  for (int k = 0; k < K; k++) delta[k] = 0;
  return 0;
}

void project_q(double* q, int K, double eps) {}
void project_p(double* p, int K, double eps) {}
