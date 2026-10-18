// The fitting loop of OpenADMIXTURE (admixture_qn!() in OpenADMIXTURE.jl,
// src/algorithms_outer.jl): a random start, a few EM iterations to prime
// the estimates, and block-wise Newton updates accelerated by the
// quasi-Newton scheme of Zhou, Alexander and Lange (2011).
//
// Why the loop is in C++
// ----------------------
// The loop was first written in R, calling the C++ kernels for each pass over
// the genotypes. Every call returned a new R matrix, and the estimates they
// replaced became garbage that R frees only when its garbage collector runs.
// With many loci each of these matrices is a copy of P, and a dozen or more
// discarded copies could accumulate before a collection, more than doubling
// the peak memory of a fit. Running the whole loop in C++ removes the cause:
// every estimate lives in a buffer allocated once at the start of the fit
// and reused at every iteration (accepting a step only swaps buffers), so a
// fit holds a fixed number of copies of the parameters, freed as soon as it
// ends. R only prepares the arguments and assembles the outputs
// (R/oadm_fit.R).
//
// The arithmetic is exactly that of the previous R loop (and of the Julia
// code): the same kernels, called in the same order, with the random start
// drawn from R's random number generator in the same order, so results are
// unchanged and reproducible with set.seed().
//
// Model and notation
// ------------------
// For individual i and locus j, the genotype g_ij (copies of the alternate
// allele, out of n_i = 2 for diploids or 1 for pseudohaploids) is binomial
// with probability qp_ij = sum_k q_ki p_kj, where q_ki is the proportion of
// the genome of individual i from ancestral population k and p_kj the
// frequency of the alternate allele at locus j in population k. We maximise
// the log-likelihood
//   L = sum_ij g_ij log(qp_ij) + (n_i - g_ij) log(1 - qp_ij)
// (over observed genotypes) subject to each column of q lying on the simplex
// and each p_kj lying in [0, 1]. q is K x I and p is K x J, column-major: the
// K values of a column, which are updated together, are contiguous.

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "geno_access.h"
#include "kernels.h"
#include "quasi_newton.h"

using namespace Rcpp;

// Format a number as R's format(x, digits = digits) does for the log
// (significant digits, scientific notation when shorter, Inf/NaN spelt out)
static std::string fmt(double x, int digits) {
  if (std::isnan(x)) return "NaN";
  if (std::isinf(x)) return x > 0 ? "Inf" : "-Inf";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*g", digits, x);
  return buf;
}

// Accumulates log lines, also reporting them as they come if verbose. They
// are reported through R's message(), as in the rest of the package, so that
// they can be silenced with suppressMessages(); this is only ever called from
// the main thread, outside parallel regions, where calling R is safe.
struct FitLog {
  bool verbose;
  std::vector<std::string> lines;
  void add(const std::string& line) {
    lines.push_back(line);
    if (verbose) {
      Function message("message");
      message(line);
    }
  }
};

// Fit the model for one value of K.
//
// geno_ptr: the genotypes (see oadm_geno_ptr()).
// k: the number of ancestral populations.
// init_q, init_p: optional starting values in the user layout (individuals
//   x K and loci x K), already validated in R; NULL to start at random.
// fix_q, fix_p: keep the corresponding block at its starting value.
// max_iter, rtol: maximum number of iterations and convergence threshold on
//   the relative change in log-likelihood.
// n_em_iter: number of EM iterations used to prime random starting values
//   (skipped when both init_q and init_p are given, as in OpenADMIXTURE.jl).
// n_qn: number of secant pairs used by the quasi-Newton step.
// Returns list(q = I x K, p = J x K, loglik, loglik_history, n_iter,
// converged, log).
// [[Rcpp::export]]
List oadm_fit_cpp(SEXP geno_ptr, int k, Nullable<NumericMatrix> init_q,
                  Nullable<NumericMatrix> init_p, bool fix_q, bool fix_p,
                  int max_iter, double rtol, int n_em_iter, int n_qn,
                  int n_cores, bool verbose) {
  const GenoView& geno = get_geno(geno_ptr);
  const size_t I = geno.nrow(), J = geno.ncol(), K = k;
  if (k < 1 || n_qn < 1 || n_cores < 1 || max_iter < 0 || n_em_iter < 0)
    stop("invalid arguments to oadm_fit_cpp()");
  const size_t nq = K * I, np = K * J;
  FitLog log = {verbose, {}};

  // Buffers, allocated once for the whole fit: the current estimate (q, p),
  // the estimates after one and two sweeps (q1, p1) and (q2, p2), and the
  // quasi-Newton point (qq, pq). Accepting a new estimate swaps buffers
  // (std::swap on vectors exchanges their contents in constant time).
  std::vector<double> q(nq), p(np), q1(nq), p1(np), q2(nq), p2(np),
    qq(nq), pq(np);

  // Random start, as in the AdmixData constructor: uniform values, with each
  // column of q rescaled to sum to one. Values are drawn from R's random
  // number generator in the same order as stats::runif() in the previous R
  // implementation (all of q, then all of p), so a given seed gives the same
  // start. (Rcpp sets up R's generator for exported functions.)
  for (size_t t = 0; t < nq; t++) q[t] = R::runif(0.0, 1.0);
  for (size_t i = 0; i < I; i++) {
    double sum = 0.0;
    for (size_t kk = 0; kk < K; kk++) sum += q[K * i + kk];
    for (size_t kk = 0; kk < K; kk++) q[K * i + kk] /= sum;
  }
  for (size_t t = 0; t < np; t++) p[t] = R::runif(0.0, 1.0);

  // EM priming, unless both blocks are given. A few iterations of the EM
  // algorithm of FRAPPE move the random start into a sensible region of the
  // parameter space; each step updates q and p simultaneously from the
  // current values (see oadm_em() in kernels.cpp).
  if (init_q.isNull() || init_p.isNull()) {
    for (int it = 1; it <= n_em_iter; it++) {
      oadm_em(geno, q.data(), p.data(), k, q1.data(), p1.data(), n_cores);
      std::swap(q, q1);
      std::swap(p, p1);
      const double ll = oadm_loglik(geno, q.data(), p.data(), k, n_cores);
      log.add("EM iteration " + std::to_string(it) + ": loglikelihood = " +
              fmt(ll, 12));
      checkUserInterrupt();
    }
  }
  // Starting values given by the user replace the corresponding block (as in
  // OpenADMIXTURE.jl, EM is still run if only one of the two is given, so
  // that the other block starts from a sensible value). They come in the
  // user layout (individuals x K, loci x K) and are transposed here.
  if (init_q.isNotNull()) {
    NumericMatrix iq(init_q.get());
    for (size_t i = 0; i < I; i++)
      for (size_t kk = 0; kk < K; kk++) q[K * i + kk] = iq(i, kk);
  }
  if (init_p.isNotNull()) {
    NumericMatrix ip(init_p.get());
    for (size_t j = 0; j < J; j++)
      for (size_t kk = 0; kk < K; kk++) p[K * j + kk] = ip(j, kk);
  }

  double ll_new = oadm_loglik(geno, q.data(), p.data(), k, n_cores);
  log.add("Initial loglikelihood: " + fmt(ll_new, 12));
  std::vector<double> ll_history(1, ll_new);

  // the last n_qn secant pairs (see quasi_newton.cpp)
  QNState qn(nq + np, n_qn);
  bool converged = false;
  int iter = 0;
  for (iter = 1; iter <= max_iter; iter++) {
    const double ll_prev = ll_new;

    // Block updates: one sweep F updates all of q given p, then all of p
    // given the new q. Given p, the log-likelihood separates into one term
    // per individual, and given q into one term per locus, so each column is
    // updated independently, by a Newton step solved exactly as a small
    // quadratic program under the constraints (simplex for q, box for p;
    // see qp_sweep.cpp). A fixed block is simply copied. We apply the sweep
    // twice: x1 = F(x0) and x2 = F(x1).
    if (fix_q) q1 = q;
    else oadm_update_q(geno, q.data(), p.data(), k, q1.data(), n_cores);
    if (fix_p) p1 = p;
    else oadm_update_p(geno, q1.data(), p.data(), k, p1.data(), n_cores);
    if (fix_q) q2 = q1;
    else oadm_update_q(geno, q1.data(), p1.data(), k, q2.data(), n_cores);
    if (fix_p) p2 = p1;
    else oadm_update_p(geno, q2.data(), p1.data(), k, p2.data(), n_cores);
    // the log-likelihood after two plain sweeps, our fallback for this
    // iteration
    const double ll_basic =
      oadm_loglik(geno, q2.data(), p2.data(), k, n_cores);

    // Quasi-Newton acceleration: the maximum likelihood estimate is a fixed
    // point of the sweep, x = F(x). Each iteration gives a secant pair
    // u = x1 - x0 and v = x2 - x1, which tells us how F changes along u; the
    // last n_qn pairs define a low-rank approximation of the Jacobian of F,
    // from which we jump towards the fixed point. The pairs are stored in a
    // circular buffer, and only those computed so far are used during the
    // first iterations. The step ignores the constraints, so qn_step()
    // projects it back onto the feasible set; it fails (and the step is
    // skipped) if the small linear system is singular.
    const size_t col = (iter - 1) % n_qn;
    qn_push(qn, col, q.data(), p.data(), q1.data(), p1.data(), q2.data(),
            p2.data(), nq, np);
    const size_t n_used = std::min((size_t) iter, (size_t) n_qn);
    double ll_qn = R_NegInf;
    if (qn_step(qn, n_used, q.data(), p.data(), q1.data(), p1.data(), k, I,
                J, qq.data(), pq.data(), n_cores))
      ll_qn = oadm_loglik(geno, qq.data(), pq.data(), k, n_cores);

    // The quasi-Newton step is a gamble: we keep it only if it improves on
    // the log-likelihood at the start of the iteration; otherwise we fall
    // back on the two plain sweeps. (As in OpenADMIXTURE.jl, the comparison
    // is with ll_prev rather than ll_basic, so an accepted step may be
    // slightly worse than x2, but it still makes progress and tends to pay
    // off over the following iterations.)
    if (std::isfinite(ll_qn) && ll_prev < ll_qn) {
      std::swap(q, qq);
      std::swap(p, pq);
      ll_new = ll_qn;
    } else {
      std::swap(q, q2);
      std::swap(p, p2);
      ll_new = ll_basic;
    }
    ll_history.push_back(ll_new);

    // Convergence: the relative change in log-likelihood over an iteration
    // falls below rtol (1e-7 by default, as in OpenADMIXTURE.jl)
    const double reldiff = std::fabs((ll_new - ll_prev) / ll_prev);
    log.add("Iteration " + std::to_string(iter) + ": loglikelihood = " +
            fmt(ll_new, 12) + ", reldiff = " + fmt(reldiff, 4) +
            " (basic = " + fmt(ll_basic, 12) + ", QN = " + fmt(ll_qn, 12) +
            ")");
    if (reldiff < rtol) {
      converged = true;
      break;
    }
    // let the user interrupt long fits (between iterations, outside any
    // parallel region; the buffers are freed automatically)
    checkUserInterrupt();
  }
  if (iter > max_iter) iter = max_iter;
  if (converged) {
    log.add("Converged in " + std::to_string(iter) + " iterations.");
  } else {
    log.add("Failed to converge after " + std::to_string(iter) +
            " iterations.");
  }
  log.add("Loglikelihood: " + fmt(ll_new, 12));

  // the estimates in the usual layout (individuals x K and loci x K)
  NumericMatrix q_out(I, K), p_out(J, K);
  for (size_t i = 0; i < I; i++)
    for (size_t kk = 0; kk < K; kk++) q_out(i, kk) = q[K * i + kk];
  for (size_t j = 0; j < J; j++)
    for (size_t kk = 0; kk < K; kk++) p_out(j, kk) = p[K * j + kk];
  return List::create(
    Named("q") = q_out, Named("p") = p_out, Named("loglik") = ll_new,
    Named("loglik_history") = wrap(ll_history), Named("n_iter") = iter,
    Named("converged") = converged, Named("log") = wrap(log.lines));
}
