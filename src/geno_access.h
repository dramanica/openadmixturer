#ifndef OADM_GENO_ACCESS_H
#define OADM_GENO_ACCESS_H

// Read-only view of a subset of a bigstatsr FBM.code256 (one raw byte per
// genotype, individuals in rows, loci in columns, column-major).
//
// Bytes are decoded with our own 256-entry lookup table (NaN = missing), so the
// result never depends on the code256 currently attached to the R object.
// Each individual has a number of trials n_i (2 for diploids, 1 for
// pseudohaploids) and a scale s_i applied to the decoded dosage (1 for
// diploids, 0.5 for pseudohaploids, which tidypopgen stores as 0/2).
//
// The view is immutable after construction and therefore safe to share
// between OpenMP threads. Do not call any R API from inside parallel regions.

#include <bigstatsr/BMAcc.h>
#include <cmath>
#include <vector>

class GenoView {
public:
  GenoView(Rcpp::Environment BM,
           const Rcpp::IntegerVector& row_ind,
           const Rcpp::IntegerVector& col_ind,
           const Rcpp::NumericVector& lut,
           const Rcpp::NumericVector& n_trials,
           const Rcpp::NumericVector& scale) {
    Rcpp::XPtr<FBM> xpBM = BM["address"];
    if (xpBM->matrix_type() != 1)
      Rcpp::stop("The backing file must be an FBM.code256 (raw bytes).");
    if (lut.size() != 256) Rcpp::stop("'lut' must have length 256.");
    size_t n = row_ind.size();
    if ((size_t) n_trials.size() != n || (size_t) scale.size() != n)
      Rcpp::stop("'n_trials' and 'scale' must have one value per individual.");
    _pMat = static_cast<const unsigned char*>(xpBM->matrix());
    _nrow_fbm = xpBM->nrow();
    // 1-based R indices -> 0-based offsets (bounds checked)
    _rows = vec_int_to_size(row_ind, xpBM->nrow(), 1);
    std::vector<size_t> cols = vec_int_to_size(col_ind, xpBM->ncol(), 1);
    _col_off.resize(cols.size());
    for (size_t j = 0; j < cols.size(); j++) _col_off[j] = cols[j] * _nrow_fbm;
    for (int b = 0; b < 256; b++) {
      _missing[b] = ISNAN(lut[b]);
      _lut[b] = _missing[b] ? 0.0 : lut[b];
    }
    _n.assign(n_trials.begin(), n_trials.end());
    _scale.assign(scale.begin(), scale.end());
  }

  size_t nrow() const { return _rows.size(); }
  size_t ncol() const { return _col_off.size(); }

  // number of trials (allele copies) for individual i
  inline double n_trials(size_t i) const { return _n[i]; }

  // Decode genotype (i, j). Returns false if missing; otherwise sets g to the
  // (scaled) alt allele count in [0, n_i].
  inline bool get(size_t i, size_t j, double& g) const {
    unsigned char b = _pMat[_rows[i] + _col_off[j]];
    if (_missing[b]) return false;
    g = _lut[b] * _scale[i];
    return true;
  }

private:
  const unsigned char* _pMat;
  size_t _nrow_fbm;
  std::vector<size_t> _rows;
  std::vector<size_t> _col_off;
  double _lut[256];
  bool _missing[256];
  std::vector<double> _n;
  std::vector<double> _scale;
};

#endif
