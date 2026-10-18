#include <Rcpp.h>
#include "geno_access.h"
using namespace Rcpp;

// Decode the selected subset into a dense matrix (NA = missing). Used only in
// tests to check that the accessor agrees with tidypopgen::show_genotypes().
// [[Rcpp::export]]
NumericMatrix oadm_decode_cpp(Environment BM,
                              const IntegerVector& rowInd,
                              const IntegerVector& colInd,
                              const NumericVector& lut,
                              const NumericVector& n_trials,
                              const NumericVector& scale) {
  GenoView geno(BM, rowInd, colInd, lut, n_trials, scale);
  size_t n = geno.nrow(), m = geno.ncol();
  NumericMatrix out(n, m);
  double g;
  for (size_t j = 0; j < m; j++)
    for (size_t i = 0; i < n; i++)
      out(i, j) = geno.get(i, j, g) ? g : NA_REAL;
  return out;
}
