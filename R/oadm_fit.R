#' Fit the OpenADMIXTURE model for a single value of K
#'
#' This is the engine behind `gt_openadmixture()`. It follows
#' `admixture_qn!()` in OpenADMIXTURE.jl: a random start (or user supplied
#' starting values), a few EM iterations to prime the estimates, and then
#' block-wise constrained Newton updates of Q and P accelerated by the
#' quasi-Newton scheme of Zhou, Alexander and Lange (2011).
#'
#' Each iteration of the main loop computes two rounds of block updates
#' (Q, then P with the new Q), giving three successive estimates x0, x1 and
#' x2. The secant pairs (x1 - x0, x2 - x1) of the last `n_qn` iterations
#' define a quasi-Newton step from x1, which is projected back onto the
#' feasible set and accepted only if it improves on the previous
#' log-likelihood; otherwise x2 is used.
#'
#' @param data the output of `oadm_prepare()`, with a `geno_ptr` element added
#'   by `oadm_attach()`
#' @param k the number of ancestral populations
#' @param init_q optional matrix (individuals x K) of starting ancestry
#'   proportions
#' @param init_p optional matrix (loci x K) of starting allele frequencies
#' @param fix_q,fix_p booleans; if `TRUE`, the corresponding parameter block
#'   is kept at its starting value
#' @param max_iter maximum number of iterations of the main loop
#' @param rtol convergence threshold on the relative change in log-likelihood
#' @param n_em_iter number of EM iterations used to prime the estimates
#'   (skipped if both `init_q` and `init_p` are given)
#' @param n_qn number of secant pairs used by the quasi-Newton step
#' @param n_cores number of threads
#' @param verbose boolean; print progress
#' @returns a list with elements `q` (individuals x K), `p` (loci x K),
#'   `loglik`, `loglik_history`, `n_iter`, `converged` and `log`
#' @keywords internal
#' @noRd
oadm_fit <- function(data, k, init_q = NULL, init_p = NULL,
                     fix_q = FALSE, fix_p = FALSE, max_iter = 1000,
                     rtol = 1e-7, n_em_iter = 5, n_qn = 3, n_cores = 1,
                     verbose = FALSE) {
  n_indiv <- data$n_indiv
  n_loci <- data$n_loci
  geno_ptr <- data$geno_ptr
  log_env <- new.env()
  log_env$lines <- character(0)
  log_msg <- function(...) {
    line <- paste0(...)
    log_env$lines <- c(log_env$lines, line)
    if (verbose) message(line)
  }

  # random start, as in the AdmixData constructor: uniform values, with each
  # column of q rescaled to sum to one
  q <- matrix(stats::runif(k * n_indiv), nrow = k)
  q <- sweep(q, 2, colSums(q), "/")
  p <- matrix(stats::runif(k * n_loci), nrow = k)

  # EM priming, unless both blocks are given
  if (is.null(init_q) || is.null(init_p)) {
    for (i in seq_len(n_em_iter)) {
      em <- oadm_em_cpp(geno_ptr, q, p, n_cores)
      q <- em$q
      p <- em$p
      log_msg(
        "EM iteration ", i, ": loglikelihood = ",
        format(oadm_loglik_cpp(geno_ptr, q, p, n_cores), digits = 12)
      )
    }
  }
  if (!is.null(init_q)) q <- t(init_q)
  if (!is.null(init_p)) p <- t(init_p)
  storage.mode(q) <- "double"
  storage.mode(p) <- "double"

  ll_new <- oadm_loglik_cpp(geno_ptr, q, p, n_cores)
  log_msg("Initial loglikelihood: ", format(ll_new, digits = 12))
  ll_history <- ll_new

  # secant pairs for the quasi-Newton step, stored as columns
  n_par <- k * (n_indiv + n_loci)
  u_mat <- matrix(0, n_par, n_qn)
  v_mat <- matrix(0, n_par, n_qn)
  converged <- FALSE
  iter <- 0
  for (iter in seq_len(max_iter)) {
    ll_prev <- ll_new
    q1 <- if (fix_q) q else oadm_update_q_cpp(geno_ptr, q, p, n_cores)
    p1 <- if (fix_p) p else oadm_update_p_cpp(geno_ptr, q1, p, n_cores)
    q2 <- if (fix_q) q1 else oadm_update_q_cpp(geno_ptr, q1, p1, n_cores)
    p2 <- if (fix_p) p1 else oadm_update_p_cpp(geno_ptr, q2, p1, n_cores)
    ll_basic <- oadm_loglik_cpp(geno_ptr, q2, p2, n_cores)

    x0 <- c(q, p)
    x1 <- c(q1, p1)
    x2 <- c(q2, p2)
    col <- (iter - 1) %% n_qn + 1
    u_mat[, col] <- x1 - x0
    v_mat[, col] <- x2 - x1
    used <- seq_len(min(iter, n_qn))
    x_qn <- oadm_qn_step(
      x0, x1, u_mat[, used, drop = FALSE],
      v_mat[, used, drop = FALSE]
    )

    ll_qn <- -Inf
    if (!is.null(x_qn)) {
      q_qn <- oadm_project_q_cpp(
        matrix(x_qn[seq_len(k * n_indiv)], nrow = k)
      )
      p_qn <- oadm_project_p_cpp(
        matrix(x_qn[k * n_indiv + seq_len(k * n_loci)], nrow = k)
      )
      ll_qn <- oadm_loglik_cpp(geno_ptr, q_qn, p_qn, n_cores)
    }
    if (is.finite(ll_qn) && ll_prev < ll_qn) {
      q <- q_qn
      p <- p_qn
      ll_new <- ll_qn
    } else {
      q <- q2
      p <- p2
      ll_new <- ll_basic
    }
    ll_history <- c(ll_history, ll_new)
    reldiff <- abs((ll_new - ll_prev) / ll_prev)
    log_msg(
      "Iteration ", iter, ": loglikelihood = ", format(ll_new, digits = 12),
      ", reldiff = ", format(reldiff, digits = 4),
      " (basic = ", format(ll_basic, digits = 12),
      ", QN = ", format(ll_qn, digits = 12), ")"
    )
    if (reldiff < rtol) {
      converged <- TRUE
      break
    }
  }
  if (converged) {
    log_msg("Converged in ", iter, " iterations.")
  } else {
    log_msg("Failed to converge after ", iter, " iterations.")
  }
  log_msg("Loglikelihood: ", format(ll_new, digits = 12))

  list(
    q = t(q),
    p = t(p),
    loglik = ll_new,
    loglik_history = ll_history,
    n_iter = iter,
    converged = converged,
    log = log_env$lines
  )
}

#' Quasi-Newton step of Zhou, Alexander and Lange (2011)
#'
#' Computes `x1 - V (U'(U - V))^-1 U'(x0 - x1)`, as `update_QN!()` in
#' OpenADMIXTURE.jl.
#' @param x0,x1 the current estimate and the estimate after one update
#' @param u_mat,v_mat matrices whose columns are the secant pairs
#' @returns the new (unprojected) estimate, or `NULL` if the linear system is
#'   singular
#' @keywords internal
#' @noRd
oadm_qn_step <- function(x0, x1, u_mat, v_mat) {
  lhs <- crossprod(u_mat, u_mat - v_mat)
  rhs <- crossprod(u_mat, x0 - x1)
  coef <- tryCatch(solve(lhs, rhs), error = function(e) NULL)
  if (is.null(coef) || any(!is.finite(coef))) {
    return(NULL)
  }
  drop(x1 - v_mat %*% coef)
}

#' Create the C++ genotype accessor for a prepared dataset
#' @param data the output of `oadm_prepare()`
#' @returns `data` with an additional `geno_ptr` element
#' @keywords internal
#' @noRd
oadm_attach <- function(data) {
  data$geno_ptr <- oadm_geno(data)
  data
}
