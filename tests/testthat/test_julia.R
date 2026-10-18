# Comparison with OpenADMIXTURE.jl
#
# Starting from the same values of Q and P, we run the internal functions of
# OpenADMIXTURE.jl (via julia/oadm_reference.jl) and the corresponding C++
# kernels of openadmixturer, and check that they agree to numerical
# precision: the log-likelihood, one EM step, one round of Newton updates of
# Q and P, and the full quasi-Newton algorithm. The data include missing
# genotypes, and more individuals and loci than the 64 x 64 tiles used by
# both implementations.
#
# These tests need Julia with OpenADMIXTURE.jl (see
# data-raw/setup_julia_env.sh, and julia_command() in helper-external.R for
# how it is found); they are skipped otherwise.

skip_on_cran()
julia <- julia_command()
skip_if(is.null(julia), "Julia with OpenADMIXTURE.jl is not available")

# simulated admixed data, written as a PLINK bed file for Julia
sim <- simulate_admixed(n = 150, m = 300, k = 4, missing = 0.03, seed = 21)
sim_gt <- make_test_gt(sim$genotypes)
work_dir <- tempfile("oadm_julia_")
dir.create(work_dir)
bed_file <- tidypopgen::gt_as_plink(sim_gt,
  file = file.path(work_dir, "sim"),
  chromosomes_as_int = TRUE
)

# starting values (K x I and K x J, the internal layout of both packages)
k <- 4
set.seed(5)
q0 <- matrix(stats::runif(k * nrow(sim_gt)), k)
q0 <- sweep(q0, 2, colSums(q0), "/")
p0 <- matrix(stats::runif(k * ncol(sim$genotypes), 0.05, 0.95), k)

# SnpArrays (and hence OpenADMIXTURE.jl) counts copies of the second allele
# of the bim file, which gt_as_plink() sets to allele_ref, whereas we count
# allele_alt: Julia's genotypes are 2 - g, and its allele frequencies 1 - p.
# The likelihood is symmetric under this change, so Q is unaffected and P
# only needs flipping.
write_csv_matrix <- function(x, name) {
  utils::write.table(x, file.path(work_dir, name),
    sep = ",",
    row.names = FALSE, col.names = FALSE
  )
}
read_csv_matrix <- function(name) {
  x <- utils::read.csv(file.path(work_dir, name), header = FALSE)
  unname(as.matrix(x))
}
write_csv_matrix(q0, "q0.csv")
write_csv_matrix(1 - p0, "p0.csv")

julia_status <- system2(
  julia$cmd,
  c(
    julia$args,
    shQuote(test_path("julia", "oadm_reference.jl")),
    shQuote(bed_file), shQuote(file.path(work_dir, "q0.csv")),
    shQuote(file.path(work_dir, "p0.csv")), shQuote(work_dir), "1000"
  ),
  stdout = FALSE, stderr = FALSE
)

dat <- oadm_attach(oadm_prepare(sim_gt))

test_that("the reference script ran", {
  expect_identical(julia_status, 0L)
})

test_that("the log-likelihood matches OpenADMIXTURE.jl", {
  expect_equal(
    oadm_loglik_cpp(dat$geno_ptr, q0, p0, 2),
    drop(read_csv_matrix("loglik.csv")),
    tolerance = 1e-12
  )
})

test_that("an EM step matches OpenADMIXTURE.jl", {
  em <- oadm_em_cpp(dat$geno_ptr, q0, p0, 2)
  # P is updated exactly as in Julia
  expect_equal(em$p, 1 - read_csv_matrix("em_p.csv"), tolerance = 1e-12)
  # Q differs by design when genotypes are missing: Julia divides by 2J, so
  # columns of Q no longer sum to one, whereas we divide by twice the number
  # of observed loci of each individual. The two differ only by the
  # normalisation of each column.
  julia_q <- read_csv_matrix("em_q.csv")
  expect_true(all(colSums(julia_q) < 1 - 1e-3))
  expect_equal(em$q, sweep(julia_q, 2, colSums(julia_q), "/"),
    tolerance = 1e-12
  )
})

test_that("Newton updates of Q and P match OpenADMIXTURE.jl", {
  q1 <- oadm_update_q_cpp(dat$geno_ptr, q0, p0, 2)
  expect_equal(q1, read_csv_matrix("upd_q.csv"), tolerance = 1e-10)
  # as in Julia, P is updated given the new Q
  p1 <- oadm_update_p_cpp(dat$geno_ptr, q1, p0, 2)
  expect_equal(p1, 1 - read_csv_matrix("upd_p.csv"), tolerance = 1e-10)
})

test_that("the full algorithm follows the same path as OpenADMIXTURE.jl", {
  fit <- oadm_fit(dat,
    k = k, init_q = t(q0), init_p = t(p0),
    n_cores = 2
  )
  julia_ll <- drop(read_csv_matrix("run_ll.csv"))
  # same number of iterations, and the same log-likelihood at each of them
  expect_true(fit$converged)
  expect_identical(fit$n_iter, length(julia_ll))
  expect_equal(fit$loglik_history[-1], julia_ll, tolerance = 1e-10)
  # same estimates
  expect_equal(fit$q, t(read_csv_matrix("run_q.csv")), tolerance = 1e-8)
  expect_equal(fit$p, t(1 - read_csv_matrix("run_p.csv")), tolerance = 1e-8)
})

unlink(work_dir, recursive = TRUE)
