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
  if (!inherits(x, "gen_tbl")) {
    stop("'x' should be a gen_tibble", call. = FALSE)
  }
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

  fbm <- attr(x$genotypes, "fbm")
  rows <- vctrs::vec_data(x$genotypes)
  cols <- tidypopgen::show_loci(x)$big_index

  # decode raw bytes ourselves rather than relying on the live code256
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
  oadm_geno_ptr(
    data$fbm, data$rows, data$cols, data$lut,
    data$n_trials, data$scale
  )
}
