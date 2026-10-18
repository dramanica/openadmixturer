# Validate against OpenADMIXTURE.jl on EUR_subset (379 x 54051), K = 4.
# Julia reports a final loglikelihood of about -1.55522e7 (24 iterations).
devtools::load_all(quiet = TRUE)
x <- tidypopgen::gt_load("dev/data/eur.gt")
for (nc in c(1, 4)) {
  t0 <- Sys.time()
  res <- gt_openadmixture(x, k = 4, seed = 7856, n_cores = nc)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat("n_cores =", nc, " time =", round(el, 1), "s  loglik =",
      format(res$loglik, digits = 12), " converged =", res$converged, "\n")
}
cat(head(res$log, 8), sep = "\n"); cat(tail(res$log, 3), sep = "\n")
q <- res$Q[[1]]
cat("rowSums range:", range(rowSums(q)), " min q:", min(q), "\n")
p <- res$P[[1]]
cat("P range:", range(p), "\n")
