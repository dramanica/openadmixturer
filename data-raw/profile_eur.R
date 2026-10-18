# Time the individual C++ kernels on EUR_subset, K = 4.
# Requires the EUR_subset data: run data-raw/download_eur_subset.R first.
# Run from the root of the package, with openadmixturer installed (timings
# from devtools::load_all() are misleading, as it compiles without
# optimisation).
library(openadmixturer)
ns <- asNamespace("openadmixturer")
x <- tidypopgen::gt_load("data-raw/data/eur.gt")
dat <- ns$oadm_attach(ns$oadm_prepare(x))
set.seed(1)
k <- 4
q <- matrix(runif(k * dat$n_indiv), k)
q <- sweep(q, 2, colSums(q), "/")
p <- matrix(runif(k * dat$n_loci), k)
for (i in 1:5) {
  em <- ns$oadm_em_cpp(dat$geno_ptr, q, p, 1)
  q <- em$q
  p <- em$p
}
for (nc in c(1, 4)) {
  cat("n_cores", nc, "\n")
  print(system.time(ns$oadm_loglik_cpp(dat$geno_ptr, q, p, nc)))
  print(system.time(ns$oadm_grad_q_cpp(dat$geno_ptr, q, p, nc)))
  print(system.time(q1 <- ns$oadm_update_q_cpp(dat$geno_ptr, q, p, nc)))
  print(system.time(ns$oadm_grad_p_cpp(dat$geno_ptr, q1, p, nc)))
  print(system.time(ns$oadm_update_p_cpp(dat$geno_ptr, q1, p, nc)))
}
