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
  # The model
  # ---------
  # For individual i and locus j, the genotype g_ij (the number of copies of
  # the alternate allele, out of n_i = 2 for diploids or 1 for pseudohaploids)
  # is binomial with probability
  #   qp_ij = sum_k q_ki p_kj,
  # where q_ki is the proportion of the genome of individual i coming from
  # ancestral population k, and p_kj is the frequency of the alternate allele
  # at locus j in population k. The log-likelihood (up to a constant) is
  #   L = sum_ij g_ij log(qp_ij) + (n_i - g_ij) log(1 - qp_ij),
  # summed over the non-missing genotypes. We maximise L subject to each
  # column of q lying on the simplex (q_ki >= 0, sum_k q_ki = 1) and each p_kj
  # lying in [0, 1].
  #
  # Storage
  # -------
  # Internally q is a K x I matrix and p a K x J matrix (one column per
  # individual and per locus), which is the layout used by the C++ kernels:
  # the K values of a column, which are updated together, are contiguous in
  # memory. They are transposed back to the usual I x K and J x K layouts
  # only when the results are returned.
  #
  # All the work on the genotypes happens in C++ (src/kernels.cpp), through
  # the accessor `geno_ptr`; this function only orchestrates the iterations,
  # whose bookkeeping is cheap (vectors of length K * (I + J)) compared with
  # the passes over the I x J genotypes.
  n_indiv <- data$n_indiv
  n_loci <- data$n_loci
  geno_ptr <- data$geno_ptr
  # the log is accumulated in an environment, so that the helper below can
  # append to it, and it is returned with the results (as with the log of
  # ADMIXTURE in tidypopgen::gt_admixture())
  log_env <- new.env()
  log_env$lines <- character(0)
  log_msg <- function(...) {
    line <- paste0(...)
    log_env$lines <- c(log_env$lines, line)
    if (verbose) message(line)
  }

  # Memory: each kernel call returns new matrices, and the previous estimates
  # become garbage. R only frees garbage when its garbage collector runs,
  # which it does when its heap reaches a threshold that grows with the
  # amount of memory in use; with many loci, a dozen or more discarded copies
  # of P could accumulate before that happened, more than doubling the peak
  # memory of a fit. So, when the parameters are large (a copy of q and p
  # takes more than about 16 MB), we ask for a minor collection after every
  # iteration: recently allocated objects are in the youngest generation, so
  # this is cheap (tens of milliseconds, against seconds for the passes over
  # the genotypes) and frees the discarded estimates straight away.
  collect_garbage <- k * (n_indiv + n_loci) > 2e6
  free_memory <- function() {
    if (collect_garbage) invisible(gc(verbose = FALSE, full = FALSE))
  }

  # random start, as in the AdmixData constructor: uniform values, with each
  # column of q rescaled to sum to one
  # (the random values are drawn from R's generator, so results are
  # reproducible with set.seed(), which gt_openadmixture() handles)
  q <- matrix(stats::runif(k * n_indiv), nrow = k)
  q <- sweep(q, 2, colSums(q), "/")
  p <- matrix(stats::runif(k * n_loci), nrow = k)

  # EM priming, unless both blocks are given
  # A few iterations of the EM algorithm of FRAPPE (Tang et al. 2005) move the
  # random start into a sensible region of the parameter space. EM steps are
  # cheap and always increase the likelihood, but converge slowly, so they are
  # only used to get started. Each step updates q and p simultaneously from
  # the current values (see oadm_em() in src/kernels.cpp for the formulas).
  if (is.null(init_q) || is.null(init_p)) {
    for (i in seq_len(n_em_iter)) {
      em <- oadm_em_cpp(geno_ptr, q, p, n_cores)
      q <- em$q
      p <- em$p
      rm(em)
      free_memory()
      log_msg(
        "EM iteration ", i, ": loglikelihood = ",
        format(oadm_loglik_cpp(geno_ptr, q, p, n_cores), digits = 12)
      )
    }
  }
  # starting values given by the user replace the corresponding block (as in
  # OpenADMIXTURE.jl, EM is still run if only one of the two is given, so that
  # the other block starts from a sensible value)
  if (!is.null(init_q)) q <- t(init_q)
  if (!is.null(init_p)) p <- t(init_p)
  # the C++ kernels expect double matrices
  storage.mode(q) <- "double"
  storage.mode(p) <- "double"

  ll_new <- oadm_loglik_cpp(geno_ptr, q, p, n_cores)
  log_msg("Initial loglikelihood: ", format(ll_new, digits = 12))
  ll_history <- ll_new

  # secant pairs for the quasi-Newton step
  # Conceptually, all parameters are stacked into a single vector x = (q, p)
  # of length K * (I + J), and the last n_qn secant pairs are kept as the
  # columns of two matrices U and V, used as a circular buffer (column `col`
  # is overwritten at each iteration). They live in C++
  # (src/quasi_newton.cpp), which works directly on the q and p blocks
  # without stacking them: in R, every vector operation would allocate a
  # new copy of the parameters, and with many loci these copies dominated the
  # memory used by a fit.
  qn_state <- oadm_qn_new_cpp(k * (n_indiv + n_loci), n_qn)
  converged <- FALSE
  iter <- 0
  for (iter in seq_len(max_iter)) {
    ll_prev <- ll_new

    # Block updates
    # -------------
    # One "sweep" F updates all of q given p, then all of p given the new q.
    # Given p, the log-likelihood separates into one term per individual, so
    # each column of q can be updated independently; likewise, given q, each
    # column of p can be updated independently. Each column update is a
    # Newton step: the log-likelihood is approximated by a quadratic around
    # the current value, and its maximum under the constraints (simplex for
    # q, box for p) is found exactly by a small quadratic program
    # (src/qp_sweep.cpp). A fixed block (fix_q or fix_p) is simply copied.
    # We apply the sweep twice: x1 = F(x0) and x2 = F(x1).
    q1 <- if (fix_q) q else oadm_update_q_cpp(geno_ptr, q, p, n_cores)
    p1 <- if (fix_p) p else oadm_update_p_cpp(geno_ptr, q1, p, n_cores)
    q2 <- if (fix_q) q1 else oadm_update_q_cpp(geno_ptr, q1, p1, n_cores)
    p2 <- if (fix_p) p1 else oadm_update_p_cpp(geno_ptr, q2, p1, n_cores)
    # the log-likelihood after two plain sweeps, our fallback for this
    # iteration
    ll_basic <- oadm_loglik_cpp(geno_ptr, q2, p2, n_cores)

    # Quasi-Newton acceleration
    # -------------------------
    # The maximum likelihood estimate is a fixed point of the sweep, x = F(x).
    # Repeatedly applying F gets there slowly, so we instead try to jump
    # directly to the fixed point. Each iteration gives a secant pair
    # u = x1 - x0 and v = x2 - x1 = F(x1) - F(x0), which tells us how F
    # changes along the direction u (v ~ dF u). The last n_qn pairs define a
    # low-rank approximation of the Jacobian dF, from which we solve for the
    # fixed point (Zhou, Alexander and Lange 2011, see src/quasi_newton.cpp).
    # Here x0 = (q, p), x1 = (q1, p1) and x2 = (q2, p2).
    col <- (iter - 1) %% n_qn + 1
    oadm_qn_push_cpp(qn_state, col, q, p, q1, p1, q2, p2)
    # during the first iterations, only the pairs computed so far are used
    n_used <- min(iter, n_qn)
    # The quasi-Newton step ignores the constraints, so it is projected back
    # onto the feasible set (each column of q onto the simplex, with a lower
    # bound of 1e-5, and each p into [1e-5, 1 - 1e-5]) before evaluating it;
    # NULL means that the small linear system was singular, and the step is
    # skipped for this iteration
    x_qn <- oadm_qn_step_cpp(qn_state, n_used, q, p, q1, p1, n_cores)
    ll_qn <- -Inf
    if (!is.null(x_qn)) {
      ll_qn <- oadm_loglik_cpp(geno_ptr, x_qn$q, x_qn$p, n_cores)
    }
    # The quasi-Newton step is a gamble: we keep it only if it improves on
    # the log-likelihood at the start of the iteration; otherwise we fall back
    # on the two plain sweeps. (As in OpenADMIXTURE.jl, the comparison is with
    # ll_prev rather than ll_basic, so an accepted step may be slightly worse
    # than x2, but it still makes progress and tends to pay off over the
    # following iterations.)
    if (is.finite(ll_qn) && ll_prev < ll_qn) {
      q <- x_qn$q
      p <- x_qn$p
      ll_new <- ll_qn
    } else {
      q <- q2
      p <- p2
      ll_new <- ll_basic
    }
    # drop the intermediate estimates now, so that their memory can be
    # reused by the next iteration (with many loci, each is a copy of P)
    rm(q1, p1, q2, p2, x_qn)
    free_memory()
    ll_history <- c(ll_history, ll_new)

    # Convergence
    # -----------
    # We stop when the relative change in log-likelihood over an iteration
    # falls below rtol (1e-7 by default, as in OpenADMIXTURE.jl).
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

  # return q and p in the usual layout (individuals x K and loci x K)
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

#' Create the C++ genotype accessor for a prepared dataset
#' @param data the output of `oadm_prepare()`
#' @returns `data` with an additional `geno_ptr` element
#' @keywords internal
#' @noRd
oadm_attach <- function(data) {
  data$geno_ptr <- oadm_geno(data)
  data
}
