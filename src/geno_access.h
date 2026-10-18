#ifndef OADM_GENO_ACCESS_H
#define OADM_GENO_ACCESS_H

// Read-only view of a subset of a bigstatsr FBM.code256 (one raw byte per
// genotype, individuals in rows, loci in columns, column-major).
//
// How tidypopgen stores genotypes
// -------------------------------
// A gen_tibble keeps its genotypes in a file-backed matrix (FBM) of type
// code256: a memory-mapped file (.bk) holding one unsigned byte per
// genotype. Rows are individuals and columns are loci, stored column-major,
// so the byte for individual r and locus c (both 0-based, in the full FBM)
// sits at offset r + c * nrow. Each byte is the count of the alternate
// allele (0, 1, 2), and other byte values encode missing data (3 for
// diploids) or imputed genotypes (4, 5, 6 after gt_impute_simple()). The R
// object carries a 256-entry table (code256) that maps bytes to values, but
// that table can be swapped at runtime (e.g. by gt_set_imputed()), so we do
// not rely on it.
//
// What this class does
// --------------------
// A gen_tibble usually shows only a subset of the FBM: some individuals may
// have been filtered out or reordered, and loci may have been selected with
// select_loci(). The view stores the selected rows and columns, so that
// get(i, j) returns the genotype of the i-th individual and the j-th locus
// of the gen_tibble, whatever their position in the FBM.
//
// Bytes are decoded with our own 256-entry lookup table (NaN = missing), so the
// result never depends on the code256 currently attached to the R object.
// Each individual has a number of trials n_i (2 for diploids, 1 for
// pseudohaploids) and a scale s_i applied to the decoded dosage (1 for
// diploids, 0.5 for pseudohaploids, which tidypopgen stores as 0/2).
// In the ADMIXTURE model, g_ij ~ Binomial(n_i, qp_ij), so a pseudohaploid
// individual contributes a single allele (0 or 1) per locus.
//
// The view is immutable after construction and therefore safe to share
// between OpenMP threads. Do not call any R API from inside parallel regions.

#include <bigstatsr/BMAcc.h>
#include <cmath>
#include <vector>

class GenoView {
public:
  // BM: the FBM (an R reference class object, i.e. an environment).
  // row_ind, col_ind: 1-based indices of the selected individuals and loci in
  //   the FBM, in the order in which they appear in the gen_tibble (they do
  //   not need to be sorted).
  // lut: 256 values, the decoded dosage of each byte (NaN for missing).
  // n_trials, scale: one value per selected individual (see above).
  // All checks happen here, in the constructor, which is always called from
  // R (single threaded), so that Rcpp::stop() can safely be used.
  GenoView(Rcpp::Environment BM,
           const Rcpp::IntegerVector& row_ind,
           const Rcpp::IntegerVector& col_ind,
           const Rcpp::NumericVector& lut,
           const Rcpp::NumericVector& n_trials,
           const Rcpp::NumericVector& scale) {
    // the "address" field of the FBM holds an external pointer to the
    // read-only memory mapping of the backing file
    Rcpp::XPtr<FBM> xpBM = BM["address"];
    // type 1 = unsigned char, i.e. code256; other FBM types (e.g. doubles)
    // cannot be decoded with a byte lookup table
    if (xpBM->matrix_type() != 1)
      Rcpp::stop("The backing file must be an FBM.code256 (raw bytes).");
    if (lut.size() != 256) Rcpp::stop("'lut' must have length 256.");
    size_t n = row_ind.size();
    if ((size_t) n_trials.size() != n || (size_t) scale.size() != n)
      Rcpp::stop("'n_trials' and 'scale' must have one value per individual.");
    // raw pointer to the first byte of the mapped file; the mapping itself is
    // kept alive by the FBM object on the R side (see oadm_geno_ptr())
    _pMat = static_cast<const unsigned char*>(xpBM->matrix());
    _nrow_fbm = xpBM->nrow();
    // 1-based R indices -> 0-based offsets (bounds checked)
    _rows = vec_int_to_size(row_ind, xpBM->nrow(), 1);
    std::vector<size_t> cols = vec_int_to_size(col_ind, xpBM->ncol(), 1);
    // precompute the offset of the start of each selected column, so that
    // get() needs a single addition instead of a multiplication
    _col_off.resize(cols.size());
    for (size_t j = 0; j < cols.size(); j++) _col_off[j] = cols[j] * _nrow_fbm;
    // split the lookup table into a missing flag and a value (0 for missing
    // bytes), so that get() never has to test for NaN
    for (int b = 0; b < 256; b++) {
      _missing[b] = ISNAN(lut[b]);
      _lut[b] = _missing[b] ? 0.0 : lut[b];
    }
    // copy the per-individual values out of R memory, so that they can be
    // read from any thread
    _n.assign(n_trials.begin(), n_trials.end());
    _scale.assign(scale.begin(), scale.end());
  }

  // number of selected individuals (I) and loci (J)
  size_t nrow() const { return _rows.size(); }
  size_t ncol() const { return _col_off.size(); }

  // number of trials (allele copies) for individual i
  inline double n_trials(size_t i) const { return _n[i]; }

  // Decode genotype (i, j). Returns false if missing; otherwise sets g to the
  // (scaled) alt allele count in [0, n_i].
  // This is the innermost operation of every kernel, so it is kept to a
  // memory read, a table lookup and a multiplication.
  inline bool get(size_t i, size_t j, double& g) const {
    unsigned char b = _pMat[_rows[i] + _col_off[j]];
    if (_missing[b]) return false;
    g = _lut[b] * _scale[i];
    return true;
  }

private:
  const unsigned char* _pMat;    // start of the memory-mapped FBM
  size_t _nrow_fbm;              // number of rows of the full FBM
  std::vector<size_t> _rows;     // 0-based FBM row of each individual
  std::vector<size_t> _col_off;  // byte offset of the start of each locus
  double _lut[256];              // decoded dosage of each byte (0 if missing)
  bool _missing[256];            // whether each byte means "missing"
  std::vector<double> _n;        // number of trials of each individual
  std::vector<double> _scale;    // dosage scale of each individual
};

#endif
