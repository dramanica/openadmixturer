#ifndef OADM_KERNELS_H
#define OADM_KERNELS_H

// Port of the CPU kernels of OpenADMIXTURE.jl (src/loops.jl and the inner
// algorithms in src/algorithms_inner.jl), reading genotypes through GenoView.
//
// All matrices are plain column-major arrays: q is K x I (column i holds the
// ancestry proportions of individual i), p is K x J (column j holds the alt
// allele frequencies of locus j), Hessians are K x K x I or K x K x J.
// Missing genotypes are skipped, and the constant 2 of the Julia code is
// replaced by the number of trials n_i of each individual (2 for diploids,
// 1 for pseudohaploids).
//
// None of these functions touches the R API, so they can be called from other
// C++ code (e.g. the outer loop) without going back to R. Results do not
// depend on the number of threads: each column is always summed in the same
// order, and there are no cross-thread reductions.

#include "geno_access.h"

// log-likelihood
double oadm_loglik(const GenoView& geno, const double* q, const double* p,
                   int K, int n_cores);

// one FRAPPE EM step (q_new and p_new are computed from the same q and p)
void oadm_em(const GenoView& geno, const double* q, const double* p, int K,
             double* q_new, double* p_new, int n_cores);

// gradient (K x I) and negative Hessian (K x K x I) of the log-likelihood
// with respect to q
void oadm_grad_q(const GenoView& geno, const double* q, const double* p,
                 int K, double* grad, double* hess, int n_cores);

// gradient (K x J) and negative Hessian (K x K x J) of the log-likelihood
// with respect to p
void oadm_grad_p(const GenoView& geno, const double* q, const double* p,
                 int K, double* grad, double* hess, int n_cores);

// one constrained Newton (QP) update of q (q_new is K x I), projected
void oadm_update_q(const GenoView& geno, const double* q, const double* p,
                   int K, double* q_new, int n_cores);

// one constrained Newton (QP) update of p (p_new is K x J), projected
void oadm_update_p(const GenoView& geno, const double* q, const double* p,
                   int K, double* p_new, int n_cores);

#endif
