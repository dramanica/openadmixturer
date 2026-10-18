// Port of OpenADMIXTURE.jl src/qp.jl (create_tableau!, quadratic_program!,
// sweep!), originally written by K. Lange and modified by S. Ko. Only the
// two cases used by update_q!/update_p! are ported: box constraints alone
// (P) and box plus the single sum constraint (Q); the p_double variant is
// not needed. Bounds are fixed at pmin = 0, pmax = 1 as in the callers.
#include "qp_sweep.h"
#include <algorithm>
#include <cmath>

// V: an orthonormal basis of R^K whose first column is ones(K) / sqrt(K),
// playing the role of svd(ones(1, K), full = true).V in Julia. We use the
// Householder reflector I - 2 u u' / (u' u), u = e1 - ones / sqrt(K), which
// maps e1 to ones / sqrt(K). It is symmetric, as is the reflector LAPACK
// returns, so V and V' are interchangeable in the mu formula below.
QPWork::QPWork(int K) : K(K), tab((K + 2) * (K + 2)), diag(K + 2),
  swept(K + 2), V(K * K), tmp(K + 2) {
  const double s = 1.0 / std::sqrt((double) K);
  std::vector<double> u(K, -s);
  u[0] += 1.0;
  double uu = 0;
  for (int k = 0; k < K; k++) uu += u[k] * u[k];
  // uu == 0 only for K == 1, where V = 1
  const double scale = uu > 0 ? 2.0 / uu : 0.0;
  for (int j = 0; j < K; j++)
    for (int i = 0; i < K; i++)
      V[i + j * K] = (i == j ? 1.0 : 0.0) - scale * u[i] * u[j];
}

// Sweep (or inverse sweep) the symmetric n x n tableau a on diagonal k.
static void sweep(double* a, int n, int k, double* tmp, bool inverse) {
  const double p = 1.0 / a[k + k * n];
  for (int i = 0; i < n; i++) {
    tmp[i] = a[i + k * n];
    a[i + k * n] = 0;
    a[k + i * n] = 0;
  }
  tmp[k] = inverse ? 1.0 : -1.0;
  for (int i = 0; i < n; i++) {
    const double pv = p * tmp[i];
    double* col = a + i * n;
    for (int j = 0; j < n; j++) col[j] -= pv * tmp[j];
  }
}

// Build the tableau as create_tableau! does. Julia is called with
// r = -grad (the caller negates Xtz, the log-likelihood gradient), and
// stores -r, so the last row/column of the tableau holds grad itself.
static int create_tableau(const double* H, const double* grad,
                          const double* par, int K, bool simplex,
                          QPWork& work) {
  const int n = simplex ? K + 2 : K + 1;
  double* tab = work.tab.data();
  const int last = n - 1;
  for (int i = 0; i < n * n; i++) tab[i] = 0;
  for (int j = 0; j < K; j++)
    for (int i = 0; i < K; i++) tab[i + j * n] = H[i + j * K];
  for (int i = 0; i < K; i++) {
    tab[last + i * n] = grad[i];
    tab[i + last * n] = grad[i];
  }
  if (!simplex) return n;
  // mu from the Gerschgorin circle theorem so that H + mu * 11' is
  // positive definite: w = V * H * V[:, 1], mu = 2 * (|w|_1 - 2 w_1) / K.
  // Deviation: Julia does not floor mu at 0, so for a well-conditioned H
  // mu can be negative and make the tableau indefinite (the original
  // Lange code, kept commented out in qp.jl, floors it at 0 and so do we).
  // In practice Julia's mu is 0 anyway: mul!(tmp_k, v, tmp_k) aliases its
  // input and output, and BLAS gemv with beta = 0 zeroes y (= x) first.
  // On the feasible set mu * (1' delta)^2 is constant, so mu only affects
  // whether the initial sweeps succeed, never the solution.
  const double* V = work.V.data();
  double* Hv = work.tmp.data();
  for (int i = 0; i < K; i++) {
    double s = 0;
    for (int j = 0; j < K; j++) s += H[i + j * K] * V[j];
    Hv[i] = s;
  }
  double norm1 = 0, w1 = 0;
  for (int i = 0; i < K; i++) {
    double s = 0;
    for (int j = 0; j < K; j++) s += V[i + j * K] * Hv[j];
    norm1 += std::fabs(s);
    if (i == 0) w1 = s;
  }
  const double mu = std::max(0.0, 2.0 * (norm1 - 2.0 * w1) / K);
  for (int j = 0; j < K; j++)
    for (int i = 0; i < K; i++) tab[i + j * n] += mu;
  for (int i = 0; i < K; i++) {
    tab[K + i * n] = 1.0;
    tab[i + K * n] = 1.0;
  }
  double sum_par = 0;
  for (int i = 0; i < K; i++) sum_par += par[i];
  tab[K + last * n] = tab[last + K * n] = 1.0 - sum_par;
  return n;
}

int qp_solve_column(const double* H, const double* grad, const double* par,
                    int K, bool simplex, double* delta, QPWork& work) {
  const double pmin = 0.0, pmax = 1.0;
  const double small = 1e-5, tol = 1e-8;
  const int p = K, c = simplex ? 1 : 0;
  const int n = create_tableau(H, grad, par, K, simplex, work);
  const int last = n - 1;
  double* tab = work.tab.data();
  double* d = work.diag.data();
  double* tmp = work.tmp.data();
  char* swept = work.swept.data();
  for (int i = 0; i < K; i++) delta[i] = 0;
  // sweep on the parameters, then on the constraint; bail out (delta = 0)
  // if the Hessian is not numerically positive definite
  for (int i = 0; i < n; i++) d[i] = tab[i + i * n];
  for (int i = 0; i < p; i++) {
    if (d[i] <= 0 || tab[i + i * n] < d[i] * tol) return 0;
    sweep(tab, n, i, tmp, false);
  }
  for (int i = 0; i < p; i++) swept[i] = 1;
  for (int i = p; i < p + c; i++) {
    if (tab[i + i * n] >= 0) return 0;
    sweep(tab, n, i, tmp, false);
  }
  for (int iteration = 1; iteration <= 1000; iteration++) {
    // maximal fractional step along the swept directions
    double a = 1.0;
    for (int i = 0; i < p; i++) {
      if (!swept[i]) continue;
      const double ui = tab[i + last * n];
      const double ai = (ui > 0 ? pmax : pmin) - par[i] - delta[i];
      if (std::fabs(ui) > 1e-10) a = std::min(a, ai / ui);
    }
    for (int i = 0; i < p; i++) {
      if (!swept[i]) continue;
      const double ui = tab[i + last * n];
      delta[i] += a * ui;
      tab[i + last * n] = (1.0 - a) * ui;
      tab[last + i * n] = tab[i + last * n];
    }
    // inverse sweep a swept parameter that has hit a bound
    bool cycle = false;
    for (int i = 0; i < p; i++) {
      const double est = par[i] + delta[i];
      const bool critical = pmin >= est - small || pmax <= est + small;
      if (swept[i] && std::fabs(tab[i + i * n]) > 1e-10 && critical) {
        sweep(tab, n, i, tmp, true);
        swept[i] = 0;
        cycle = true;
        break;
      }
    }
    if (cycle) continue;
    // sweep an unswept parameter that violates the KKT conditions
    for (int i = 0; i < p; i++) {
      const double ui = tab[i + last * n];
      const double est = par[i] + delta[i];
      const bool violation = (ui > 0 && pmin >= est - small) ||
        (ui < 0 && pmax <= est + small);
      if (!swept[i] && violation) {
        sweep(tab, n, i, tmp, false);
        swept[i] = 1;
        cycle = true;
        break;
      }
    }
    if (cycle) continue;
    return iteration;
  }
  return 0;
}
