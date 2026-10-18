#ifndef OADM_PROJECTIONS_H
#define OADM_PROJECTIONS_H

// Port of OpenADMIXTURE.jl src/projections.jl.
//
// project_q: Euclidean projection of q[0..K-1] onto the shifted simplex
//   {sum(q) == 1, q >= eps}, in place.
// project_p: clamp p[0..K-1] to [eps, 1 - eps], in place.
// Both are thread-safe (no shared state, no R API).

const double OADM_EPS = 1e-5;

void project_q(double* q, int K, double eps = OADM_EPS);
void project_p(double* p, int K, double eps = OADM_EPS);

#endif
