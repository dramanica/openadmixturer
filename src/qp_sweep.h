#ifndef OADM_QP_SWEEP_H
#define OADM_QP_SWEEP_H

// Port of OpenADMIXTURE.jl src/qp.jl: K. Lange's sweep-operator active-set
// quadratic program, used for the per-column Newton steps of Q and P.
//
// For one column (an individual for Q, a locus for P) with current value
// par[0..K-1], gradient of the log-likelihood grad[0..K-1] and NEGATIVE
// Hessian H (K x K, column-major, symmetric positive semi-definite), find
//   delta = argmin 0.5 * delta' H delta - grad' delta
// subject to 0 <= par + delta <= 1 and, if simplex is true,
//   sum(delta) == 1 - sum(par).
// On failure delta is left as far as the solver got (0 if it failed during
// the initial sweeps), exactly as in the Julia code.
//
// QPWork holds all scratch memory for a given K; create one per thread.
// Nothing here touches the R API, so it is safe inside OpenMP regions.

#include <vector>

struct QPWork {
  int K;
  std::vector<double> tab;     // tableau, (K+2) x (K+2) column-major max
  std::vector<double> diag;    // saved diagonal
  std::vector<char> swept;     // swept flags
  std::vector<double> V;       // K x K right singular vectors of ones(1, K)
  std::vector<double> tmp;     // length-K scratch
  explicit QPWork(int K);
};

// returns the number of active-set iterations used, or 0 on failure
int qp_solve_column(const double* H, const double* grad, const double* par,
                    int K, bool simplex, double* delta, QPWork& work);

#endif
