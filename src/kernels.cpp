// Port of the CPU kernels of OpenADMIXTURE.jl (src/loops.jl and the inner
// algorithms in src/algorithms_inner.jl). See kernels.h for conventions.
//
// Work is organised in tiles of TILE_I individuals x TILE_J loci. Q-side work
// (sums over loci for each individual) is parallelised over blocks of TILE_I
// individuals, P-side work (sums over individuals for each locus) over blocks
// of TILE_J loci. Each thread owns the output columns of its blocks, so there
// are no races and no cross-thread reductions, and every column is summed in
// the same order whatever the number of threads.
//
// The innermost unit of work is a slice of one locus x up to TILE_I
// individuals (individuals are contiguous in the column-major FBM). For each
// slice, qp is computed once into a thread-local buffer with unit-stride loops
// over a transposed copy of q, the genotypes are decoded, and the likelihood
// coefficients a = g / qp, b = (n - g) / (1 - qp) (and wa = a / qp,
// wb = b / (1 - qp) for Hessians) are computed without branches, so that all
// these loops can be vectorised. The accumulation loops then run over the
// slice:
// - Q side: accumulators are stored as K (and K (K + 1) / 2 for the upper
//   triangle of the Hessian) vectors of length TILE_I, updated with
//   unit-stride vectorisable loops for every locus;
// - P side: each accumulator gets a (SIMD) dot product over the slice; the
//   TILE_I x K corner of q stays in L1 while the TILE_J loci are processed.

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "geno_access.h"
#include "kernels.h"
#include "qp_sweep.h"
#include "projections.h"
using namespace Rcpp;

static const size_t TILE_I = 64;
static const size_t TILE_J = 64;

// number of blocks of size `tile` needed to cover n
static inline size_t n_blocks(size_t n, size_t tile) {
  return (n + tile - 1) / tile;
}

// copy the upper triangle (k2 >= k, stored in column k) to the lower one
static inline void mirror_hess(double* H, int K) {
  for (int k = 0; k < K; k++)
    for (int k2 = k + 1; k2 < K; k2++) H[k + K * k2] = H[k2 + K * k];
}

static std::vector<double> one_minus(const double* p, size_t n) {
  std::vector<double> omp(n);
  for (size_t x = 0; x < n; x++) omp[x] = 1.0 - p[x];
  return omp;
}

// q is K x I; returns its transpose (I x K, column-major)
static std::vector<double> transpose(const double* q, int K, size_t I) {
  std::vector<double> qT((size_t) K * I);
  for (size_t i = 0; i < I; i++)
    for (int k = 0; k < K; k++) qT[i + I * k] = q[k + K * i];
  return qT;
}

// Thread-local buffers for one slice (one locus x up to TILE_I individuals)
struct Slice {
  std::vector<double> qp, g, h, a, b, wa, wb, tmp;
  Slice() : qp(TILE_I), g(TILE_I), h(TILE_I), a(TILE_I), b(TILE_I),
    wa(TILE_I), wb(TILE_I), tmp(TILE_I) {}
};

// qp and decoded genotypes for locus j and individuals [i0, i1). Missing
// genotypes get g = h = 0 (h = n - g), so they drop out of every sum.
static inline void slice_geno(const GenoView& geno, const double* qT,
                              const double* pj, int K, size_t j, size_t i0,
                              size_t i1, Slice& s) {
  const size_t I = geno.nrow(), ni = i1 - i0;
  double* __restrict qp = s.qp.data();
  double* __restrict g = s.g.data();
  double* __restrict h = s.h.data();
  std::fill(qp, qp + ni, 0.0);
  for (int k = 0; k < K; k++) {
    const double pk = pj[k];
    const double* __restrict qk = qT + I * k + i0;
#pragma omp simd
    for (size_t ii = 0; ii < ni; ii++) qp[ii] += qk[ii] * pk;
  }
  double gij;
  for (size_t ii = 0; ii < ni; ii++) {
    if (geno.get(i0 + ii, j, gij)) {
      g[ii] = gij;
      h[ii] = geno.n_trials(i0 + ii) - gij;
    } else {
      g[ii] = 0.0;
      h[ii] = 0.0;
    }
  }
}

// Likelihood coefficients for the slice. A term whose coefficient (g or
// n - g) is zero is set to 0 (dividing by 1 instead of qp or 1 - qp), so that
// qp of exactly 0 or 1 does not produce NaN.
template <bool HESS>
static inline void slice_coef(const GenoView& geno, const double* qT,
                              const double* pj, int K, size_t j, size_t i0,
                              size_t i1, Slice& s) {
  slice_geno(geno, qT, pj, K, j, i0, i1, s);
  const size_t ni = i1 - i0;
  const double* __restrict qp = s.qp.data();
  const double* __restrict g = s.g.data();
  const double* __restrict h = s.h.data();
  double* __restrict a = s.a.data();
  double* __restrict b = s.b.data();
  double* __restrict wa = s.wa.data();
  double* __restrict wb = s.wb.data();
#pragma omp simd
  for (size_t ii = 0; ii < ni; ii++) {
    // x = (g > 0) ? qp : 1 and y = (h > 0) ? 1 - qp : 1, written with
    // arithmetic masks so that the loop has no control flow
    const double mg = (double) (g[ii] > 0), mh = (double) (h[ii] > 0);
    const double x = mg * qp[ii] + (1.0 - mg);
    const double y = mh * (1.0 - qp[ii]) + (1.0 - mh);
    a[ii] = g[ii] / x;
    b[ii] = h[ii] / y;
    if (HESS) {
      wa[ii] = a[ii] / x;
      wb[ii] = b[ii] / y;
    }
  }
}

// Q side: for individuals [i0, i1), sum over all loci
//   grad_k = sum_j a p_kj + b (1 - p_kj)
//   hess_{k,k2} = sum_j wa p_kj p_k2j + wb (1 - p_kj)(1 - p_k2j)   (if HESS)
// grad is K x ni and hess K x K x ni (ni = i1 - i0); both are overwritten.
// gT and hT are scratch accumulators of size K * TILE_I and
// K (K + 1) / 2 * TILE_I. omp = 1 - p.
template <bool HESS>
static void accum_q_block(const GenoView& geno, const double* qT,
                          const double* p, const double* omp, int K,
                          size_t i0, size_t i1, double* grad, double* hess,
                          Slice& s, double* gT, double* hT) {
  const size_t J = geno.ncol(), ni = i1 - i0, KK = (size_t) K * K;
  const size_t n_pair = (size_t) K * (K + 1) / 2;
  std::fill(gT, gT + K * TILE_I, 0.0);
  if (HESS) std::fill(hT, hT + n_pair * TILE_I, 0.0);
  const double* __restrict a = s.a.data();
  const double* __restrict b = s.b.data();
  const double* __restrict wa = s.wa.data();
  const double* __restrict wb = s.wb.data();
  for (size_t j = 0; j < J; j++) {
    const double* pj = p + K * j;
    const double* oj = omp + K * j;
    slice_coef<HESS>(geno, qT, pj, K, j, i0, i1, s);
    for (int k = 0; k < K; k++) {
      const double pk = pj[k], ok = oj[k];
      double* __restrict gk = gT + TILE_I * k;
#pragma omp simd
      for (size_t ii = 0; ii < ni; ii++) gk[ii] += a[ii] * pk + b[ii] * ok;
    }
    if (HESS) {
      double* __restrict hk = hT;
      for (int k = 0; k < K; k++) {
        for (int k2 = k; k2 < K; k2++, hk += TILE_I) {
          const double c1 = pj[k] * pj[k2], c2 = oj[k] * oj[k2];
#pragma omp simd
          for (size_t ii = 0; ii < ni; ii++)
            hk[ii] += wa[ii] * c1 + wb[ii] * c2;
        }
      }
    }
  }
  for (size_t ii = 0; ii < ni; ii++) {
    for (int k = 0; k < K; k++) grad[k + K * ii] = gT[ii + TILE_I * k];
    if (HESS) {
      double* H = hess + KK * ii;
      size_t idx = 0;
      for (int k = 0; k < K; k++)
        for (int k2 = k; k2 < K; k2++, idx++)
          H[k2 + K * k] = hT[ii + TILE_I * idx];
      mirror_hess(H, K);
    }
  }
}

// P side: for loci [j0, j1), sum over all individuals. Two modes:
// - EM = false: out1 = grad (K x nj), grad_k = sum_i (a - b) q_ki;
//               out2 = hess (K x K x nj),
//               hess_{k,k2} = sum_i (wa + wb) q_ki q_k2i
// - EM = true:  out1 = sum_i a q_ki and out2 = sum_i b q_ki (both K x nj)
// Outputs are overwritten.
template <bool EM>
static void accum_p_block(const GenoView& geno, const double* qT,
                          const double* p, int K, size_t j0, size_t j1,
                          double* out1, double* out2, Slice& s) {
  const size_t I = geno.nrow(), nj = j1 - j0, KK = (size_t) K * K;
  std::fill(out1, out1 + K * nj, 0.0);
  std::fill(out2, out2 + (EM ? K : KK) * nj, 0.0);
  double* __restrict a = s.a.data();
  double* __restrict b = s.b.data();
  double* __restrict wa = s.wa.data();
  const double* __restrict wb = s.wb.data();
  double* __restrict wq = s.tmp.data();
  for (size_t i0 = 0; i0 < I; i0 += TILE_I) {
    const size_t i1 = std::min(I, i0 + TILE_I), ni = i1 - i0;
    for (size_t jj = 0; jj < nj; jj++) {
      const size_t j = j0 + jj;
      slice_coef<!EM>(geno, qT, p + K * j, K, j, i0, i1, s);
      double* o1 = out1 + K * jj;
      double* o2 = out2 + (EM ? K : KK) * jj;
      if (EM) {
        for (int k = 0; k < K; k++) {
          const double* __restrict qk = qT + I * k + i0;
          double sa = 0.0, sb = 0.0;
#pragma omp simd reduction(+:sa, sb)
          for (size_t ii = 0; ii < ni; ii++) {
            sa += a[ii] * qk[ii];
            sb += b[ii] * qk[ii];
          }
          o1[k] += sa;
          o2[k] += sb;
        }
      } else {
        // a <- a - b and wa <- wa + wb
#pragma omp simd
        for (size_t ii = 0; ii < ni; ii++) {
          a[ii] -= b[ii];
          wa[ii] += wb[ii];
        }
        for (int k = 0; k < K; k++) {
          const double* __restrict qk = qT + I * k + i0;
          double sg = 0.0;
#pragma omp simd reduction(+:sg)
          for (size_t ii = 0; ii < ni; ii++) sg += a[ii] * qk[ii];
          o1[k] += sg;
#pragma omp simd
          for (size_t ii = 0; ii < ni; ii++) wq[ii] = wa[ii] * qk[ii];
          double* Hk = o2 + K * k;
          for (int k2 = k; k2 < K; k2++) {
            const double* __restrict qk2 = qT + I * k2 + i0;
            double sh = 0.0;
#pragma omp simd reduction(+:sh)
            for (size_t ii = 0; ii < ni; ii++) sh += wq[ii] * qk2[ii];
            Hk[k2] += sh;
          }
        }
      }
    }
  }
  if (!EM)
    for (size_t jj = 0; jj < nj; jj++) mirror_hess(out2 + KK * jj, K);
}

// x^e for e in {0, 1, 2} (Lagrange interpolation in e), without control flow;
// gives 1 for e = 0 even if x = 0
static inline double pow012(double x, double e) {
  return 0.5 * (e - 1.0) * (e - 2.0) + x * e * (2.0 - e) +
    0.5 * x * x * e * (e - 1.0);
}

// Products of up to LOG_BATCH factors of at least LOG_MIN_Z cannot underflow
// (LOG_BATCH^2 must equal TILE_I)
static const int LOG_BATCH = 8;
static const double LOG_MIN_Z = 1e-30;

// Log-likelihood of one slice (after slice_geno). Each observed genotype
// contributes g log(x) + h log(y), with x = (g > 0) ? qp : 1 and
// y = (h > 0) ? 1 - qp : 1 (h = n - g). For hard calls g and h are 0, 1 or 2,
// so the contribution is log(z) with z = qp^g (1 - qp)^h; the z of LOG_BATCH
// genotypes are multiplied together before taking a single log, which makes
// this several times cheaper than one or two logs per genotype. Any other
// genotype (or one with a z so small that the product could underflow) is
// added with the direct formula.
static double loglik_slice(size_t ni, Slice& s) {
  const double* __restrict qp = s.qp.data();
  const double* __restrict g = s.g.data();
  const double* __restrict h = s.h.data();
  double* __restrict z = s.a.data();
  double* __restrict c = s.b.data();
#pragma omp simd
  for (size_t ii = 0; ii < ni; ii++) {
    z[ii] = pow012(qp[ii], g[ii]) * pow012(1.0 - qp[ii], h[ii]);
    // c == 0 iff g and h are both 0, 1 or 2
    c[ii] = std::fabs(g[ii] * (g[ii] - 1.0) * (g[ii] - 2.0)) +
      std::fabs(h[ii] * (h[ii] - 1.0) * (h[ii] - 2.0));
  }
  // the exceptions are rare, so this branch is almost always predicted
  double ll = 0.0;
  for (size_t ii = 0; ii < ni; ii++) {
    if ((z[ii] < LOG_MIN_Z) | (c[ii] != 0)) {
      z[ii] = 1.0;
      if (g[ii] > 0) ll += g[ii] * std::log(qp[ii]);
      if (h[ii] > 0) ll += h[ii] * std::log(1.0 - qp[ii]);
    }
  }
  // LOG_BATCH interleaved products of LOG_BATCH factors each (the slice is
  // padded with ones), so that the products vectorise too
  std::fill(z + ni, z + TILE_I, 1.0);
  double prod[LOG_BATCH];
  for (int b = 0; b < LOG_BATCH; b++) prod[b] = 1.0;
  for (int m = 0; m < LOG_BATCH; m++)
#pragma omp simd
    for (int b = 0; b < LOG_BATCH; b++) prod[b] *= z[LOG_BATCH * m + b];
  for (int b = 0; b < LOG_BATCH; b++) ll += std::log(prod[b]);
  return ll;
}

double oadm_loglik(const GenoView& geno, const double* q, const double* p,
                   int K, int n_cores) {
  const size_t I = geno.nrow(), J = geno.ncol();
  std::vector<double> qT = transpose(q, K, I);
  const ptrdiff_t nb = n_blocks(J, TILE_J);
  // one partial sum per block, added up in order: deterministic
  std::vector<double> partial(nb, 0.0);
#pragma omp parallel num_threads(n_cores)
{
  Slice s;
#pragma omp for schedule(dynamic)
  for (ptrdiff_t bl = 0; bl < nb; bl++) {
    const size_t j1 = std::min(J, (bl + 1) * TILE_J);
    double ll = 0.0;
    for (size_t j = bl * TILE_J; j < j1; j++) {
      for (size_t i0 = 0; i0 < I; i0 += TILE_I) {
        const size_t i1 = std::min(I, i0 + TILE_I);
        slice_geno(geno, qT.data(), p + K * j, K, j, i0, i1, s);
        ll += loglik_slice(i1 - i0, s);
      }
    }
    partial[bl] = ll;
  }
}
  double ll = 0.0;
  for (ptrdiff_t bl = 0; bl < nb; bl++) ll += partial[bl];
  return ll;
}

// Two passes from the same (q, p): a Q-side pass over blocks of individuals
// and a P-side pass over blocks of loci. This reads the genotypes twice but
// keeps both passes free of reductions.
void oadm_em(const GenoView& geno, const double* q, const double* p, int K,
             double* q_new, double* p_new, int n_cores) {
  const size_t I = geno.nrow(), J = geno.ncol();
  std::vector<double> omp = one_minus(p, K * J);
  std::vector<double> qT = transpose(q, K, I);
  const ptrdiff_t nbi = n_blocks(I, TILE_I), nbj = n_blocks(J, TILE_J);
#pragma omp parallel num_threads(n_cores)
{
  Slice s;
  std::vector<double> gT(K * TILE_I);
  // Q: q_new_ki = q_ki * sum_j (a p_kj + b (1 - p_kj)) / (n_i J_i), where
  // J_i is the number of loci observed in individual i. The denominator equals
  // the column sum of the numerator, so we normalise by that: this keeps q on
  // the simplex with missing data (Julia divides by n_i J, which does not).
#pragma omp for schedule(dynamic)
  for (ptrdiff_t bl = 0; bl < nbi; bl++) {
    const size_t i0 = bl * TILE_I, i1 = std::min(I, i0 + TILE_I);
    accum_q_block<false>(geno, qT.data(), p, omp.data(), K, i0, i1,
                         q_new + K * i0, NULL, s, gT.data(), NULL);
    for (size_t i = i0; i < i1; i++) {
      double denom = 0.0;
      for (int k = 0; k < K; k++) {
        q_new[k + K * i] *= q[k + K * i];
        denom += q_new[k + K * i];
      }
      if (denom > 0)
        for (int k = 0; k < K; k++) q_new[k + K * i] /= denom;
      else
        for (int k = 0; k < K; k++) q_new[k + K * i] = q[k + K * i];
    }
  }
  // P: p_new = A / (A + B), A = p sum_i a q, B = (1 - p) sum_i b q
  std::vector<double> sa(K * TILE_J), sb(K * TILE_J);
#pragma omp for schedule(dynamic)
  for (ptrdiff_t bl = 0; bl < nbj; bl++) {
    const size_t j0 = bl * TILE_J, j1 = std::min(J, j0 + TILE_J);
    accum_p_block<true>(geno, qT.data(), p, K, j0, j1, sa.data(), sb.data(),
                        s);
    for (size_t j = j0; j < j1; j++) {
      for (int k = 0; k < K; k++) {
        const size_t x = k + K * j, y = k + K * (j - j0);
        const double A = p[x] * sa[y], B = omp[x] * sb[y];
        p_new[x] = (A + B > 0) ? A / (A + B) : p[x];
      }
    }
  }
}
}

void oadm_grad_q(const GenoView& geno, const double* q, const double* p,
                 int K, double* grad, double* hess, int n_cores) {
  const size_t I = geno.nrow(), J = geno.ncol(), KK = (size_t) K * K;
  std::vector<double> omp = one_minus(p, K * J);
  std::vector<double> qT = transpose(q, K, I);
  const ptrdiff_t nb = n_blocks(I, TILE_I);
#pragma omp parallel num_threads(n_cores)
{
  Slice s;
  std::vector<double> gT(K * TILE_I), hT(K * (K + 1) / 2 * TILE_I);
#pragma omp for schedule(dynamic)
  for (ptrdiff_t bl = 0; bl < nb; bl++) {
    const size_t i0 = bl * TILE_I, i1 = std::min(I, i0 + TILE_I);
    accum_q_block<true>(geno, qT.data(), p, omp.data(), K, i0, i1,
                        grad + K * i0, hess + KK * i0, s, gT.data(),
                        hT.data());
  }
}
}

void oadm_grad_p(const GenoView& geno, const double* q, const double* p,
                 int K, double* grad, double* hess, int n_cores) {
  const size_t I = geno.nrow(), J = geno.ncol(), KK = (size_t) K * K;
  std::vector<double> qT = transpose(q, K, I);
  const ptrdiff_t nb = n_blocks(J, TILE_J);
#pragma omp parallel num_threads(n_cores)
{
  Slice s;
#pragma omp for schedule(dynamic)
  for (ptrdiff_t bl = 0; bl < nb; bl++) {
    const size_t j0 = bl * TILE_J, j1 = std::min(J, j0 + TILE_J);
    accum_p_block<false>(geno, qT.data(), p, K, j0, j1, grad + K * j0,
                         hess + KK * j0, s);
  }
}
}

// For each block, the gradient and Hessian are accumulated into thread-local
// buffers and the per-column QPs are solved straight away, so the full
// K x K x I Hessian is never stored.
void oadm_update_q(const GenoView& geno, const double* q, const double* p,
                   int K, double* q_new, int n_cores) {
  const size_t I = geno.nrow(), J = geno.ncol(), KK = (size_t) K * K;
  std::vector<double> omp = one_minus(p, K * J);
  std::vector<double> qT = transpose(q, K, I);
  const ptrdiff_t nb = n_blocks(I, TILE_I);
#pragma omp parallel num_threads(n_cores)
{
  Slice s;
  std::vector<double> gT(K * TILE_I), hT(K * (K + 1) / 2 * TILE_I);
  std::vector<double> grad(K * TILE_I), hess(KK * TILE_I), delta(K);
  QPWork work(K);
#pragma omp for schedule(dynamic)
  for (ptrdiff_t bl = 0; bl < nb; bl++) {
    const size_t i0 = bl * TILE_I, i1 = std::min(I, i0 + TILE_I);
    accum_q_block<true>(geno, qT.data(), p, omp.data(), K, i0, i1,
                        grad.data(), hess.data(), s, gT.data(), hT.data());
    for (size_t i = i0; i < i1; i++) {
      const size_t ii = i - i0;
      const double* qi = q + K * i;
      qp_solve_column(&hess[KK * ii], &grad[K * ii], qi, K, true,
                      delta.data(), work);
      double* qn = q_new + K * i;
      for (int k = 0; k < K; k++) qn[k] = qi[k] + delta[k];
      project_q(qn, K);
    }
  }
}
}

void oadm_update_p(const GenoView& geno, const double* q, const double* p,
                   int K, double* p_new, int n_cores) {
  const size_t I = geno.nrow(), J = geno.ncol(), KK = (size_t) K * K;
  std::vector<double> qT = transpose(q, K, I);
  const ptrdiff_t nb = n_blocks(J, TILE_J);
#pragma omp parallel num_threads(n_cores)
{
  Slice s;
  std::vector<double> grad(K * TILE_J), hess(KK * TILE_J), delta(K);
  QPWork work(K);
#pragma omp for schedule(dynamic)
  for (ptrdiff_t bl = 0; bl < nb; bl++) {
    const size_t j0 = bl * TILE_J, j1 = std::min(J, j0 + TILE_J);
    accum_p_block<false>(geno, qT.data(), p, K, j0, j1, grad.data(),
                         hess.data(), s);
    for (size_t j = j0; j < j1; j++) {
      const size_t jj = j - j0;
      const double* pj = p + K * j;
      qp_solve_column(&hess[KK * jj], &grad[K * jj], pj, K, false,
                      delta.data(), work);
      double* pn = p_new + K * j;
      for (int k = 0; k < K; k++) pn[k] = pj[k] + delta[k];
      project_p(pn, K);
    }
  }
}
}

// ---------------------------------------------------------------------------
// R interface

static SEXP geno_tag() {
  return Rf_install("oadm_geno_view");
}

// Build the genotype accessor once; every kernel then reuses it. The FBM
// environment is protected by the external pointer, so the memory mapping
// outlives the pointer even if R drops its own references to the FBM.
// [[Rcpp::export]]
SEXP oadm_geno_ptr(Environment BM, IntegerVector rowInd, IntegerVector colInd,
                   NumericVector lut, NumericVector n_trials,
                   NumericVector scale) {
  GenoView* geno = new GenoView(BM, rowInd, colInd, lut, n_trials, scale);
  XPtr<GenoView> ptr(geno, true, geno_tag(), BM);
  return ptr;
}

static const GenoView& get_geno(SEXP geno_ptr) {
  if (TYPEOF(geno_ptr) != EXTPTRSXP || R_ExternalPtrTag(geno_ptr) != geno_tag())
    stop("'geno_ptr' must be created by oadm_geno_ptr().");
  XPtr<GenoView> ptr(geno_ptr);
  if (ptr.get() == NULL)
    stop("'geno_ptr' is no longer valid (was it saved and reloaded?).");
  return *ptr;
}

// check q (K x I) and p (K x J) against the genotypes; returns K
static int check_dims(const GenoView& geno, const NumericMatrix& q,
                      const NumericMatrix& p, int n_cores) {
  const int K = q.nrow();
  if (K < 1) stop("'q' must have at least one row.");
  if (p.nrow() != K) stop("'q' and 'p' must have the same number of rows (K).");
  if ((size_t) q.ncol() != geno.nrow())
    stop("'q' must have one column per individual (%d), not %d.",
         (int) geno.nrow(), q.ncol());
  if ((size_t) p.ncol() != geno.ncol())
    stop("'p' must have one column per locus (%d), not %d.",
         (int) geno.ncol(), p.ncol());
  if (n_cores < 1) stop("'n_cores' must be at least 1.");
  return K;
}

// [[Rcpp::export]]
double oadm_loglik_cpp(SEXP geno_ptr, NumericMatrix q, NumericMatrix p,
                       int n_cores) {
  const GenoView& geno = get_geno(geno_ptr);
  const int K = check_dims(geno, q, p, n_cores);
  return oadm_loglik(geno, q.begin(), p.begin(), K, n_cores);
}

// [[Rcpp::export]]
List oadm_em_cpp(SEXP geno_ptr, NumericMatrix q, NumericMatrix p,
                 int n_cores) {
  const GenoView& geno = get_geno(geno_ptr);
  const int K = check_dims(geno, q, p, n_cores);
  NumericMatrix q_new(K, q.ncol()), p_new(K, p.ncol());
  oadm_em(geno, q.begin(), p.begin(), K, q_new.begin(), p_new.begin(),
          n_cores);
  return List::create(_["q"] = q_new, _["p"] = p_new);
}

// [[Rcpp::export]]
List oadm_grad_q_cpp(SEXP geno_ptr, NumericMatrix q, NumericMatrix p,
                     int n_cores) {
  const GenoView& geno = get_geno(geno_ptr);
  const int K = check_dims(geno, q, p, n_cores);
  const int I = q.ncol();
  NumericMatrix grad(K, I);
  NumericVector hess((size_t) K * K * I);
  hess.attr("dim") = IntegerVector::create(K, K, I);
  oadm_grad_q(geno, q.begin(), p.begin(), K, grad.begin(), hess.begin(),
              n_cores);
  return List::create(_["grad"] = grad, _["hess"] = hess);
}

// [[Rcpp::export]]
List oadm_grad_p_cpp(SEXP geno_ptr, NumericMatrix q, NumericMatrix p,
                     int n_cores) {
  const GenoView& geno = get_geno(geno_ptr);
  const int K = check_dims(geno, q, p, n_cores);
  const int J = p.ncol();
  NumericMatrix grad(K, J);
  NumericVector hess((size_t) K * K * J);
  hess.attr("dim") = IntegerVector::create(K, K, J);
  oadm_grad_p(geno, q.begin(), p.begin(), K, grad.begin(), hess.begin(),
              n_cores);
  return List::create(_["grad"] = grad, _["hess"] = hess);
}

// [[Rcpp::export]]
NumericMatrix oadm_update_q_cpp(SEXP geno_ptr, NumericMatrix q,
                                NumericMatrix p, int n_cores) {
  const GenoView& geno = get_geno(geno_ptr);
  const int K = check_dims(geno, q, p, n_cores);
  NumericMatrix q_new(K, q.ncol());
  oadm_update_q(geno, q.begin(), p.begin(), K, q_new.begin(), n_cores);
  return q_new;
}

// [[Rcpp::export]]
NumericMatrix oadm_update_p_cpp(SEXP geno_ptr, NumericMatrix q,
                                NumericMatrix p, int n_cores) {
  const GenoView& geno = get_geno(geno_ptr);
  const int K = check_dims(geno, q, p, n_cores);
  NumericMatrix p_new(K, p.ncol());
  oadm_update_p(geno, q.begin(), p.begin(), K, p_new.begin(), n_cores);
  return p_new;
}
