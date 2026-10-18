// Rcpp wrappers around the QP solver and the projections. The QP wrapper is
// used only in tests; oadm_project_q_cpp and oadm_project_p_cpp are also
// used by the R fitting loop for the quasi-Newton step.
//
// In more detail: the Newton updates themselves call qp_solve_column,
// project_q and project_p directly from C++ (src/kernels.cpp), so none of
// these wrappers sits on the hot path. R/oadm_fit.R uses the two projection
// wrappers to bring the quasi-Newton extrapolated point back into the
// feasible set before its log-likelihood is evaluated, and
// R/gt_openadmixture.R uses oadm_project_q_cpp on user-supplied starting
// values. oadm_qp_solve_cpp exists so that the solver can be checked
// against small hand-made problems from testthat.
#include <Rcpp.h>
#include "qp_sweep.h"
#include "projections.h"
using namespace Rcpp;

// Solve one column QP (see qp_sweep.h for the problem statement).
// H: K x K negative Hessian; grad, par: length-K gradient and current
// column; simplex: TRUE for a Q column (sum constraint), FALSE for P.
// Returns list(delta = step, iter = active-set iterations, 0 on failure).
// [[Rcpp::export]]
List oadm_qp_solve_cpp(const NumericMatrix& H, const NumericVector& grad,
                       const NumericVector& par, bool simplex) {
  const int K = grad.size();
  // the C++ solver does no checking of its own, so validate shapes here
  if (H.nrow() != K || H.ncol() != K || par.size() != K)
    stop("H must be K x K and grad, par of length K");
  // a fresh scratch object per call; fine for testing, the kernels reuse
  // one per thread instead
  QPWork work(K);
  NumericVector delta(K);
  // R matrices are column-major, as the solver expects, so the data
  // pointers can be passed straight through
  int iter = qp_solve_column(H.begin(), grad.begin(), par.begin(), K,
                             simplex, delta.begin(), work);
  return List::create(Named("delta") = delta, Named("iter") = iter);
}

// Project each column of q onto {sum == 1, entries >= 1e-5}; returns a copy.
// q is K x I (one column per individual, the layout used internally).
// [[Rcpp::export]]
NumericMatrix oadm_project_q_cpp(const NumericMatrix& q) {
  // clone so that the caller's matrix is not modified in place
  NumericMatrix out = clone(q);
  const int K = out.nrow(), n = out.ncol();
  double* x = out.begin();
  // columns are contiguous in column-major storage; the size_t cast keeps
  // the offset from overflowing int for large I
  for (int i = 0; i < n; i++) project_q(x + (size_t) i * K, K);
  return out;
}

// Clamp every entry of p to [1e-5, 1 - 1e-5]; returns a copy.
// p is K x J (one column per locus, the layout used internally).
//
// Clamping is entry-wise, so we could clamp the whole matrix in one call,
// and the first version did exactly that, passing out.size() as the length.
// But project_p() takes an int (it was written for a single column of K
// values, as in Julia), and out.size() is an R_xlen_t: with more than
// 2^31 - 1 entries (e.g. K = 10 and over 215 million loci) the length would
// overflow and the call would clamp the wrong number of entries. Clamping
// one column at a time, as project_q() does above, keeps every length equal
// to K, and the size_t cast keeps the offset from overflowing. (This
// wrapper is specific to this package, so there is no Julia counterpart.)
// [[Rcpp::export]]
NumericMatrix oadm_project_p_cpp(const NumericMatrix& p) {
  NumericMatrix out = clone(p);
  const int K = out.nrow(), n = out.ncol();
  double* x = out.begin();
  for (int j = 0; j < n; j++) project_p(x + (size_t) j * K, K);
  return out;
}
