devtools::load_all(quiet = TRUE)
x <- tidypopgen::gt_load("dev/data/eur.gt")
dat <- oadm_attach(oadm_prepare(x))
set.seed(1); k <- 4
q <- matrix(runif(k * dat$n_indiv), k); q <- sweep(q, 2, colSums(q), "/")
p <- matrix(runif(k * dat$n_loci), k)
for (i in 1:5) { em <- oadm_em_cpp(dat$geno_ptr, q, p, 1); q <- em$q; p <- em$p }
for (nc in c(1, 4)) {
  cat("n_cores", nc, "\n")
  print(system.time(oadm_loglik_cpp(dat$geno_ptr, q, p, nc)))
  print(system.time(oadm_grad_q_cpp(dat$geno_ptr, q, p, nc)))
  print(system.time(q1 <- oadm_update_q_cpp(dat$geno_ptr, q, p, nc)))
  print(system.time(oadm_grad_p_cpp(dat$geno_ptr, q1, p, nc)))
  print(system.time(oadm_update_p_cpp(dat$geno_ptr, q1, p, nc)))
}
