// Rcpp wrappers around the QP solver and the projections. The QP wrapper is
// used only in tests; oadm_project_q_cpp and oadm_project_p_cpp are also
// used by the R fitting loop for the quasi-Newton step.
#include <Rcpp.h>
#include "qp_sweep.h"
#include "projections.h"
using namespace Rcpp;

// Solve one column QP (see qp_sweep.h for the problem statement).
// [[Rcpp::export]]
List oadm_qp_solve_cpp(const NumericMatrix& H, const NumericVector& grad,
                       const NumericVector& par, bool simplex) {
  const int K = grad.size();
  if (H.nrow() != K || H.ncol() != K || par.size() != K)
    stop("H must be K x K and grad, par of length K");
  QPWork work(K);
  NumericVector delta(K);
  int iter = qp_solve_column(H.begin(), grad.begin(), par.begin(), K,
                             simplex, delta.begin(), work);
  return List::create(Named("delta") = delta, Named("iter") = iter);
}

// Project each column of q onto {sum == 1, entries >= 1e-5}; returns a copy.
// [[Rcpp::export]]
NumericMatrix oadm_project_q_cpp(const NumericMatrix& q) {
  NumericMatrix out = clone(q);
  const int K = out.nrow(), n = out.ncol();
  double* x = out.begin();
  for (int i = 0; i < n; i++) project_q(x + (size_t) i * K, K);
  return out;
}

// Clamp every entry of p to [1e-5, 1 - 1e-5]; returns a copy.
// [[Rcpp::export]]
NumericMatrix oadm_project_p_cpp(const NumericMatrix& p) {
  NumericMatrix out = clone(p);
  project_p(out.begin(), out.size());
  return out;
}
