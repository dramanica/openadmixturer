// Quasi-Newton acceleration of Zhou, Alexander and Lange (2011), as in
// update_UV!() and update_QN!() of OpenADMIXTURE.jl (src/quasi_newton.jl).
//
// The fitting loop (R/oadm_fit.R) applies the block-update map F twice per
// iteration, giving x0 (the current estimate), x1 = F(x0) and x2 = F(x1),
// where x = (q, p) stacks all the parameters. The last n_qn secant pairs
// u = x1 - x0 and v = x2 - x1 are kept as the columns of U and V, and the
// quasi-Newton step jumps towards the fixed point of F:
//   x_qn = x1 - V (U'(U - V))^-1 U'(x0 - x1).
//
// Derivation. We look for the fixed point of F. Near the current estimate,
// F(x) ~ F(x0) + M (x - x0), where M approximates the Jacobian of F. The
// secant pairs require M U = V (each column of V is the change in F caused by
// the change in x in the corresponding column of U), and the smallest such
// matrix is M = V (U'U)^-1 U'. Solving x = F(x0) + M (x - x0) with this M
// (using the Woodbury identity to avoid ever forming the large matrix M)
// gives the formula above, with x1 = F(x0). Only an n_qn x n_qn system needs
// to be solved, so the step is cheap even with millions of parameters.
//
// Why this is in C++
// ------------------
// The step only involves vectors of length K (I + J) and an n_qn x n_qn
// linear system (3 x 3 by default), so it is cheap; it was first written in
// R. But in R every operation allocates: stacking q and p into x0, x1 and x2,
// the differences x1 - x0 and x2 - x1, U - V inside crossprod(), and so on.
// With many loci, each of these is a copy of P, and since R frees memory
// only when the garbage collector runs, the peak memory of a fit reached
// about 30 copies of the parameters. Here U and V are kept in a single
// object for the whole fit, and the step is computed directly from the Q and
// P blocks, without stacking them and without temporary vectors: the only
// new allocation is the projected result. This is purely an implementation
// change: the arithmetic is the same as in the Julia code (up to rounding,
// as sums are accumulated in a different order).
//
// Determinism: the long sums (inner products over K (I + J) values) are
// split into fixed chunks whose partial sums are added up in order, so the
// result does not depend on the number of threads.

#include <RcppArmadillo.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include "projections.h"

using namespace Rcpp;

// The secant pairs: U and V are n_par x n_qn, column-major, with each column
// laid out as x = (q, p): the K x I values of q followed by the K x J values
// of p, both column-major (the internal layout of the fitting loop).
struct QNState {
  size_t n_par, n_qn;
  std::vector<double> U, V;
  QNState(size_t n_par, size_t n_qn)
    : n_par(n_par), n_qn(n_qn), U(n_par * n_qn, 0.0), V(n_par * n_qn, 0.0) {}
};

// number of values per chunk of the long sums (fixed, so that results do not
// depend on the number of threads)
static const size_t QN_CHUNK = 65536;

static QNState& get_state(SEXP state) {
  XPtr<QNState> ptr(state);
  if (ptr.get() == NULL)
    stop("the quasi-Newton state is no longer valid");
  return *ptr;
}

// Create the secant pairs for a fit with n_par = K (I + J) parameters,
// keeping the last n_qn pairs.
// [[Rcpp::export]]
SEXP oadm_qn_new_cpp(double n_par, int n_qn) {
  if (n_par < 1 || n_qn < 1) stop("'n_par' and 'n_qn' must be positive");
  XPtr<QNState> ptr(new QNState((size_t) n_par, (size_t) n_qn), true);
  return ptr;
}

// Store the secant pair of this iteration in column `col` (1-based) of U and
// V: u = x1 - x0 and v = x2 - x1, with x0 = (q, p), x1 = (q1, p1) and
// x2 = (q2, p2). The columns are reused cyclically by the caller, so U and V
// always hold the last n_qn pairs.
// [[Rcpp::export]]
void oadm_qn_push_cpp(SEXP state, int col, const NumericMatrix& q,
                  const NumericMatrix& p, const NumericMatrix& q1,
                  const NumericMatrix& p1, const NumericMatrix& q2,
                  const NumericMatrix& p2) {
  QNState& st = get_state(state);
  const size_t nq = q.size(), np = p.size();
  if (nq + np != st.n_par || (size_t) q1.size() != nq ||
      (size_t) q2.size() != nq || (size_t) p1.size() != np ||
      (size_t) p2.size() != np)
    stop("the dimensions of q and p do not match the quasi-Newton state");
  if (col < 1 || (size_t) col > st.n_qn) stop("'col' is out of range");
  double* u = &st.U[(size_t) (col - 1) * st.n_par];
  double* v = &st.V[(size_t) (col - 1) * st.n_par];
  // the q block, then the p block
  for (size_t t = 0; t < nq; t++) {
    u[t] = q1[t] - q[t];
    v[t] = q2[t] - q1[t];
  }
  for (size_t t = 0; t < np; t++) {
    u[nq + t] = p1[t] - p[t];
    v[nq + t] = p2[t] - p1[t];
  }
}

// The quasi-Newton step from x1 = (q1, p1), using the first n_used columns of
// U and V, projected back onto the feasible set (each column of q onto the
// simplex, p into [1e-5, 1 - 1e-5]). Returns list(q, p), or NULL if the
// n_used x n_used system is (numerically) singular, in which case the caller
// skips the quasi-Newton step for this iteration.
// [[Rcpp::export]]
SEXP oadm_qn_step_cpp(SEXP state, int n_used, const NumericMatrix& q,
                  const NumericMatrix& p, const NumericMatrix& q1,
                  const NumericMatrix& p1, int n_cores) {
  QNState& st = get_state(state);
  const size_t nq = q.size(), np = p.size(), n_par = st.n_par;
  if (nq + np != n_par || (size_t) q1.size() != nq ||
      (size_t) p1.size() != np)
    stop("the dimensions of q and p do not match the quasi-Newton state");
  if (n_used < 1 || (size_t) n_used > st.n_qn) stop("'n_used' is out of range");
  if (n_cores < 1) stop("'n_cores' must be at least 1");
  const size_t m = n_used;
  // raw pointers, so that nothing in the parallel region touches R objects
  const double *pq = q.begin(), *pp = p.begin(), *pq1 = q1.begin(),
    *pp1 = p1.begin();
  const double *U = st.U.data(), *V = st.V.data();

  // The small system: lhs = U'(U - V) (m x m) and rhs = U'(x0 - x1) (m),
  // i.e. for each pair of columns (a, b), sum_t U[t, a] (U[t, b] - V[t, b]),
  // and for each column a, sum_t U[t, a] (x0[t] - x1[t]). These are computed
  // in one pass over the parameters, chunk by chunk, with the partial sums
  // of each chunk stored separately and added up in chunk order afterwards.
  const size_t n_chunks = (n_par + QN_CHUNK - 1) / QN_CHUNK;
  const size_t n_sums = m * m + m;
  std::vector<double> partial(n_chunks * n_sums, 0.0);
#pragma omp parallel for num_threads(n_cores) schedule(static)
  for (ptrdiff_t c = 0; c < (ptrdiff_t) n_chunks; c++) {
    const size_t t0 = c * QN_CHUNK, t1 = std::min(n_par, t0 + QN_CHUNK);
    double* sums = &partial[c * n_sums];
    for (size_t t = t0; t < t1; t++) {
      // x0 - x1 at position t, read from the q or the p block
      const double d = t < nq ? pq[t] - pq1[t] : pp[t - nq] - pp1[t - nq];
      for (size_t a = 0; a < m; a++) {
        const double ua = U[a * n_par + t];
        for (size_t b = 0; b < m; b++)
          sums[a + m * b] += ua * (U[b * n_par + t] - V[b * n_par + t]);
        sums[m * m + a] += ua * d;
      }
    }
  }
  arma::mat lhs(m, m, arma::fill::zeros);
  arma::vec rhs(m, arma::fill::zeros);
  for (size_t c = 0; c < n_chunks; c++) {
    const double* sums = &partial[c * n_sums];
    for (size_t a = 0; a < m; a++) {
      for (size_t b = 0; b < m; b++) lhs(a, b) += sums[a + m * b];
      rhs(a) += sums[m * m + a];
    }
  }

  // Solve the small system. As with solve() in R (which the first version of
  // this code used), a system that is singular to machine precision is
  // rejected: this happens when secant pairs are (nearly) collinear, e.g.
  // close to convergence, and the step is then skipped.
  if (!lhs.is_finite() || !rhs.is_finite()) return R_NilValue;
  const double rc = arma::rcond(lhs);
  if (!(rc >= DBL_EPSILON)) return R_NilValue;
  arma::vec coef;
  if (!arma::solve(coef, lhs, rhs, arma::solve_opts::no_approx))
    return R_NilValue;
  if (!coef.is_finite()) return R_NilValue;

  // x_qn = x1 - V coef, written straight into the new q and p matrices, then
  // projected column by column (columns of K values)
  const int K = q.nrow();
  NumericMatrix q_new(K, q.ncol()), p_new(K, p.ncol());
  double *qn = q_new.begin(), *pn = p_new.begin();
  std::vector<double> cf(coef.begin(), coef.end());
#pragma omp parallel for num_threads(n_cores) schedule(static)
  for (ptrdiff_t c = 0; c < (ptrdiff_t) n_chunks; c++) {
    const size_t t0 = c * QN_CHUNK, t1 = std::min(n_par, t0 + QN_CHUNK);
    for (size_t t = t0; t < t1; t++) {
      double step = 0.0;
      for (size_t a = 0; a < m; a++) step += V[a * n_par + t] * cf[a];
      if (t < nq) qn[t] = pq1[t] - step;
      else pn[t - nq] = pp1[t - nq] - step;
    }
  }
  const size_t n_indiv = q.ncol(), n_loci = p.ncol();
#pragma omp parallel for num_threads(n_cores) schedule(static)
  for (ptrdiff_t i = 0; i < (ptrdiff_t) n_indiv; i++)
    project_q(qn + (size_t) i * K, K);
#pragma omp parallel for num_threads(n_cores) schedule(static)
  for (ptrdiff_t j = 0; j < (ptrdiff_t) n_loci; j++)
    project_p(pn + (size_t) j * K, K);
  return List::create(Named("q") = q_new, Named("p") = p_new);
}
