#' Prepare a gen_tibble for the OpenADMIXTURE kernels
#'
#' Extracts the FBM, the row (individual) and column (locus) indices of the
#' current selection, the lookup table used to decode the raw bytes, and the
#' number of trials (allele copies) per individual.
#'
#' @param x a [`tidypopgen::gen_tibble`]
#' @param use_imputed boolean; if `TRUE`, imputed genotypes (bytes 4-6 in
#'   bigsnpr's `CODE_IMPUTE_PRED`) are used as observed genotypes; otherwise
#'   they are treated as missing.
#' @returns a list with elements `fbm`, `rows`, `cols`, `lut`, `n_trials`,
#'   `scale`, `n_indiv` and `n_loci`
#' @keywords internal
#' @noRd
oadm_prepare <- function(x, use_imputed = FALSE) {
  # Everything the C++ code needs is gathered here, on the R side, so that the
  # kernels only deal with plain vectors and a pointer to the backing file.
  if (!inherits(x, "gen_tbl")) {
    stop("'x' should be a gen_tibble", call. = FALSE)
  }
  # The likelihood is binomial with n_i = 2 (diploids) or n_i = 1
  # (pseudohaploids). tidypopgen codes a gen_tibble's ploidy as 2 (diploid),
  # -2 (pseudohaploid, i.e. a mix of diploids and pseudohaploids), 0 (mixed
  # ploidy) or k (uniform polyploid); only the first two are supported.
  ploidy <- tidypopgen::show_ploidy(x)
  if (!ploidy %in% c(2, -2)) {
    stop(
      "gt_openadmixture() only works with diploid or pseudohaploid data ",
      "(ploidy 2 or -2); this gen_tibble has ploidy ", ploidy,
      call. = FALSE
    )
  }
  if (use_imputed && !tidypopgen::gt_has_imputed(x)) {
    stop("'use_imputed = TRUE' but this gen_tibble has not been imputed",
      call. = FALSE
    )
  }

  # The genotypes column of a gen_tibble is a vctrs vector whose data are the
  # (1-based) rows of the FBM for each individual, and which carries the FBM
  # as an attribute. The loci table records the FBM column of each locus in
  # `big_index`. Filtering, reordering and select_loci() only change these
  # indices, never the backing file, so we pass them on to C++; they can be
  # in any order.
  fbm <- attr(x$genotypes, "fbm")
  rows <- vctrs::vec_data(x$genotypes)
  cols <- tidypopgen::show_loci(x)$big_index

  # decode raw bytes ourselves rather than relying on the live code256
  # Bytes 0, 1 and 2 are the counts of the alternate allele (R indices 1:3
  # correspond to bytes 0:2). Imputed genotypes are stored as bytes 4, 5 and
  # 6 (bigsnpr's CODE_IMPUTE_PRED), and are only used if requested. Every
  # other byte (including 3, the code for missing data) maps to NA, which the
  # C++ code treats as missing.
  lut <- rep(NA_real_, 256)
  lut[1:3] <- 0:2
  if (use_imputed) {
    lut[5:7] <- 0:2
  }

  # pseudohaploid individuals are stored as 0/2 but carry a single allele
  indiv_ploidy <- tidypopgen::indiv_ploidy(x)
  if (!all(indiv_ploidy %in% c(1, 2))) {
    stop("individual ploidies should be 1 (pseudohaploid) or 2",
      call. = FALSE
    )
  }
  # n_trials is the number of allele copies of each individual (n_i in the
  # binomial likelihood); scale converts the stored dosage into a count out
  # of n_i (a stored 2 becomes 1 for a pseudohaploid individual)
  n_trials <- as.numeric(indiv_ploidy)
  scale <- n_trials / 2

  list(
    fbm = fbm,
    rows = as.integer(rows),
    cols = as.integer(cols),
    lut = lut,
    n_trials = n_trials,
    scale = scale,
    n_indiv = length(rows),
    n_loci = length(cols)
  )
}

#' Decode the genotypes selected in a prepared dataset into a dense matrix
#'
#' Mostly useful for testing.
#' @param data the output of `oadm_prepare()`
#' @returns a numeric matrix (individuals x loci), `NA` for missing values
#' @keywords internal
#' @noRd
oadm_decode <- function(data) {
  oadm_decode_cpp(
    data$fbm, data$rows, data$cols, data$lut,
    data$n_trials, data$scale
  )
}

#' Create the genotype accessor used by the C++ kernels
#'
#' The accessor is built once and reused by every kernel call. It keeps the
#' FBM alive, but it does not survive saving and reloading the R session.
#' @param data the output of `oadm_prepare()`
#' @returns an external pointer
#' @keywords internal
#' @noRd
oadm_geno <- function(data) {
  # The pointer wraps a C++ GenoView (src/geno_access.h), which holds the
  # memory-mapped backing file and the decoding tables. Building it once
  # avoids re-checking and copying the indices at every kernel call (there
  # are several kernel calls per iteration).
  oadm_geno_ptr(
    data$fbm, data$rows, data$cols, data$lut,
    data$n_trials, data$scale
  )
}
