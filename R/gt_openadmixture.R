#' Estimate ancestry proportions with the OpenADMIXTURE algorithm
#'
#' This function estimates individual ancestry proportions (Q) and the allele
#' frequencies of the ancestral populations (P) under the ADMIXTURE model,
#' using the algorithm of OpenADMIXTURE.jl (Ko et al. 2023): EM priming,
#' block-wise Newton updates solved as constrained quadratic programs, and
#' quasi-Newton acceleration. Genotypes are read directly from the backing
#' file of the `gen_tibble` (respecting any filtering of individuals and
#' selection of loci), so no intermediate PLINK file is written. Computation
#' is multithreaded with OpenMP.
#'
#' @details Missing genotypes are ignored in the likelihood. Pseudohaploid
#'   individuals (see [tidypopgen::gt_pseudohaploid()]) contribute a single
#'   allele per locus. Only diploid and pseudohaploid `gen_tibble`s are
#'   supported.
#'
#'   `P` gives the frequency of the **alternate** allele (`allele_alt` in
#'   [tidypopgen::show_loci()]) in each ancestral population.
#'
#'   Supplying `init_p` together with `fix_p = TRUE` estimates the ancestry of
#'   new individuals given the allele frequencies of a reference panel (a
#'   "projection" analysis); the loci of `x` must then match the rows of
#'   `init_p`.
#'
#' @references Ko S, Chu BB, Peterson D, Okenwa C, Papp JC, Alexander DH,
#'   Sobel EM, Zhou H, Lange KL (2023) Unsupervised discovery of ancestry
#'   informative markers and genetic admixture proportions in biobank-scale
#'   datasets. The American Journal of Human Genetics, 110(2), 314-325.
#'
#'   Alexander DH, Novembre J, Lange K (2009) Fast model-based estimation of
#'   ancestry in unrelated individuals. Genome Research, 19(9), 1655-1664.
#'
#' @param x a [`tidypopgen::gen_tibble`] (possibly grouped)
#' @param k an integer vector giving the number(s) of ancestral populations
#' @param n_runs the number of runs for each value of `k` (defaults to 1)
#' @param seed the seed for the random number generator (defaults to `NULL`).
#'   Either a vector of length `n_runs` (the same seeds are used for each `k`)
#'   or of length `n_runs * length(k)`. The global random number generator
#'   state is restored on exit.
#' @param max_iter the maximum number of iterations (defaults to 1000)
#' @param rtol the convergence threshold on the relative change in
#'   log-likelihood between iterations (defaults to 1e-7)
#' @param n_em_iter the number of EM iterations used to prime the estimates
#'   (defaults to 5)
#' @param n_qn the number of secant pairs used by the quasi-Newton
#'   acceleration (defaults to 3)
#' @param init_q optional matrix (individuals x k) of starting ancestry
#'   proportions; only allowed with a single value of `k`
#' @param init_p optional matrix (loci x k) of starting allele frequencies;
#'   only allowed with a single value of `k`
#' @param fix_q,fix_p booleans; if `TRUE`, Q (respectively P) is kept at the
#'   value given in `init_q` (respectively `init_p`)
#' @param use_imputed boolean; if `TRUE`, imputed genotypes are used as if
#'   they were observed; otherwise they are treated as missing (the default)
#' @param n_cores the number of cores to use (defaults to 1)
#' @param verbose boolean; should progress be printed (defaults to `FALSE`)
#' @returns an object of class `gt_admix`, a list with elements:
#' - `k` the number of clusters of each run
#' - `Q` a list of [tidypopgen::q_matrix()] objects with the ancestry
#'   proportions
#' - `P` a list of matrices (loci x k) with the alternate allele frequencies
#' - `log` a list with the log of each run
#' - `loglik` the log-likelihood of each run
#' - `converged` whether each run converged
#' - `id` the id column of `x`
#' - `group` the group column of `x` (if `x` is grouped)
#' - `algorithm` "OpenADMIXTURE"
#' @export
#' @examples
#' bed_file <-
#'   system.file("extdata", "pop_a.bed", package = "tidypopgen")
#' pop_a <- tidypopgen::gen_tibble(bed_file,
#'   backingfile = tempfile("pop_a"),
#'   quiet = TRUE
#' )
#' adm <- gt_openadmixture(pop_a, k = 2, seed = 1)
#' adm$Q[[1]]
gt_openadmixture <- function(x, k, n_runs = 1, seed = NULL, max_iter = 1000,
                             rtol = 1e-7, n_em_iter = 5, n_qn = 3,
                             init_q = NULL, init_p = NULL, fix_q = FALSE,
                             fix_p = FALSE, use_imputed = FALSE, n_cores = 1,
                             verbose = FALSE) {
  # This function is a thin wrapper: it checks the arguments, prepares the
  # data once, and then calls oadm_fit() (R/oadm_fit.R) for each run,
  # collecting the results into a gt_admix object with the same structure as
  # the one returned by tidypopgen::gt_admixture().

  # Check the arguments
  k <- as.integer(k)
  n_runs <- as.integer(n_runs)
  if (length(k) == 0 || any(is.na(k)) || any(k < 1)) {
    stop("'k' should be a vector of positive integers", call. = FALSE)
  }
  if (length(n_runs) != 1 || is.na(n_runs) || n_runs < 1) {
    stop("'n_runs' should be a positive integer", call. = FALSE)
  }
  n_seeds_ok <- length(seed) %in% c(n_runs, n_runs * length(k))
  if (!is.null(seed) && !n_seeds_ok) {
    stop(
      "'seed' should be a vector of length 'n_runs' OR 'n_runs' * ",
      "'length(k)'",
      call. = FALSE
    )
  }
  # as in gt_admixture(), seeds can be given per run (and reused for each k)
  # or per combination of k and run
  if (!is.null(seed) && length(seed) == n_runs) {
    seed <- rep(seed, length(k))
  }
  if ((!is.null(init_q) || !is.null(init_p)) && length(k) > 1) {
    stop("'init_q' and 'init_p' can only be used with a single 'k'",
      call. = FALSE
    )
  }
  # fixing both blocks would leave nothing to estimate (and make the
  # quasi-Newton system singular); fixing a block without starting values
  # would fix it at a random value
  if (fix_q && fix_p) {
    stop("'fix_q' and 'fix_p' cannot both be TRUE", call. = FALSE)
  }
  if (fix_q && is.null(init_q)) {
    stop("'fix_q = TRUE' requires 'init_q'", call. = FALSE)
  }
  if (fix_p && is.null(init_p)) {
    stop("'fix_p = TRUE' requires 'init_p'", call. = FALSE)
  }

  # Prepare the data: indices into the backing file, decoding table and
  # ploidy of each individual (R/oadm_data.R). This is done once and shared
  # by all runs.
  data <- oadm_prepare(x, use_imputed = use_imputed)
  if (any(k >= data$n_indiv)) {
    stop("'k' should be smaller than the number of individuals",
      call. = FALSE
    )
  }
  init_q <- oadm_check_init(init_q, data$n_indiv, k[1], "init_q", TRUE)
  init_p <- oadm_check_init(init_p, data$n_loci, k[1], "init_p", FALSE)
  # build the C++ accessor to the genotypes, reused by every kernel call
  data <- oadm_attach(data)

  # use our own seeds without disturbing the global RNG state
  # (the random starting values are drawn with R's generator, so set.seed()
  # makes each run reproducible; on exit we restore the user's RNG state, or
  # remove .Random.seed if there was none before)
  if (!is.null(seed)) {
    if (exists(".Random.seed", envir = globalenv())) {
      old_seed <- get(".Random.seed", envir = globalenv())
      on.exit(assign(".Random.seed", old_seed, envir = globalenv())) # nolint
    } else {
      on.exit(rm(".Random.seed", envir = globalenv()))
    }
  }

  # Run the fits, one per combination of k and run, filling in the elements
  # of the gt_admix object in the same order as gt_admixture() (by k, then by
  # run)
  loci_names <- tidypopgen::show_loci(x)$name
  adm_list <- list(
    k = integer(0), Q = list(), P = list(), log = list(),
    loglik = numeric(0), converged = logical(0)
  )
  index <- 1
  for (this_k in k) {
    for (this_rep in seq_len(n_runs)) {
      if (!is.null(seed)) set.seed(seed[index])
      if (verbose) {
        message("Running k = ", this_k, ", run ", this_rep)
      }
      fit <- oadm_fit(
        data,
        k = this_k, init_q = init_q, init_p = init_p,
        fix_q = fix_q, fix_p = fix_p, max_iter = max_iter, rtol = rtol,
        n_em_iter = n_em_iter, n_qn = n_qn, n_cores = n_cores,
        verbose = verbose
      )
      # P has one row per locus, in the order of show_loci(x), and gives the
      # frequency of the alternate allele in each ancestral population
      p_mat <- fit$p
      dimnames(p_mat) <- list(loci_names, paste0(".P", seq_len(this_k)))
      adm_list$k[index] <- this_k
      # q_matrix() gives Q the class and column names (.Q1, .Q2, ...) that
      # tidypopgen's methods (autoplot(), tidy(), augment()) expect
      adm_list$Q[[index]] <- tidypopgen::q_matrix(fit$q)
      adm_list$P[[index]] <- p_mat
      adm_list$log[[index]] <- fit$log
      adm_list$loglik[index] <- fit$loglik
      adm_list$converged[index] <- fit$converged
      index <- index + 1
    }
  }

  # metadata used by tidypopgen's methods to label and group individuals in
  # plots and tables
  adm_list$id <- x$id
  if (inherits(x, "grouped_gen_tbl")) {
    adm_list$group <- x[[dplyr::group_vars(x)]]
  }
  adm_list$algorithm <- "OpenADMIXTURE"
  class(adm_list) <- c("gt_admix", "list")
  adm_list
}

#' Validate a matrix of starting values
#' @param init the matrix (or `NULL`)
#' @param n_rows the expected number of rows
#' @param k the expected number of columns
#' @param arg the name of the argument, for error messages
#' @param simplex boolean; should rows sum to one
#' @returns a numeric matrix, or `NULL`
#' @keywords internal
#' @noRd
oadm_check_init <- function(init, n_rows, k, arg, simplex) {
  if (is.null(init)) {
    return(NULL)
  }
  init <- as.matrix(init)
  storage.mode(init) <- "double"
  if (nrow(init) != n_rows || ncol(init) != k) {
    stop("'", arg, "' should be a matrix with ", n_rows, " rows and ", k,
      " columns",
      call. = FALSE
    )
  }
  if (any(!is.finite(init)) || any(init < 0) || any(init > 1)) {
    stop("'", arg, "' should only contain values between 0 and 1",
      call. = FALSE
    )
  }
  if (simplex && any(abs(rowSums(init) - 1) > 1e-6)) {
    stop("the rows of '", arg, "' should sum to 1", call. = FALSE)
  }
  # keep starting values strictly inside the feasible set
  # (the same bounds used for the estimates: entries of Q of at least 1e-5,
  # with rows summing to one, and P in [1e-5, 1 - 1e-5]); this avoids
  # log(0) in the log-likelihood and keeps the Newton steps well defined
  eps <- 1e-5
  if (simplex) {
    init <- t(oadm_project_q_cpp(t(init)))
  } else {
    init <- pmin(pmax(init, eps), 1 - eps)
  }
  unname(init)
}
