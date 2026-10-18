# Helpers for the tests that validate openadmixturer against the original
# implementations: finding Julia with OpenADMIXTURE.jl (test_julia.R), and
# downloading the EUR_subset data compared with stored ADMIXTURE results
# (test_admixture.R). When Julia or the data are not available, the
# corresponding tests are skipped, so that the test suite still runs on CRAN
# and on CI machines without them.

# Locate the conda executable, if any
find_conda <- function() {
  candidates <- c(
    Sys.getenv("CONDA_EXE"),
    Sys.which("conda"),
    path.expand(file.path(
      "~", c("miniconda3", "anaconda3", "miniforge3", "mambaforge"),
      "bin", "conda"
    ))
  )
  if (requireNamespace("reticulate", quietly = TRUE)) {
    candidates <- c(
      candidates,
      tryCatch(reticulate::conda_binary(), error = function(e) "")
    )
  }
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  if (length(candidates) == 0) {
    return(NULL)
  }
  candidates[[1]]
}

# Command used to run Julia with OpenADMIXTURE.jl, as a list with the
# executable (`cmd`) and the arguments that precede the script (`args`), or
# NULL if no suitable installation is found. In order of preference:
# - the executable in the environment variable OADM_JULIA (whose default
#   environment must include OpenADMIXTURE.jl);
# - the conda environment created by data-raw/setup_julia_env.sh (its name
#   can be changed with OADM_JULIA_ENV), run through `conda run` so that its
#   Julia depot is used;
# - julia on the PATH.
# The candidate is accepted only if it can load OpenADMIXTURE.jl. Julia is
# started with --threads=1,0 (one thread, no interactive thread pool), as
# OpenADMIXTURE.jl indexes its per-thread buffers with threadid(), which
# breaks with the interactive thread added by default since Julia 1.12; a
# single thread also makes its results deterministic.
julia_command <- function() {
  julia_args <- "--threads=1,0"
  candidates <- list()
  if (nzchar(Sys.getenv("OADM_JULIA"))) {
    candidates <- c(candidates, list(list(
      cmd = Sys.getenv("OADM_JULIA"), args = julia_args
    )))
  }
  conda <- find_conda()
  if (!is.null(conda)) {
    env <- Sys.getenv("OADM_JULIA_ENV", "openadmixturer_julia")
    candidates <- c(candidates, list(list(
      cmd = conda,
      args = c("run", "--no-capture-output", "-n", env, "julia", julia_args)
    )))
  }
  if (nzchar(Sys.which("julia"))) {
    candidates <- c(candidates, list(list(
      cmd = Sys.which("julia"), args = julia_args
    )))
  }
  for (candidate in candidates) {
    status <- suppressWarnings(system2(
      candidate$cmd,
      c(candidate$args, "-e", shQuote("using OpenADMIXTURE")),
      stdout = FALSE, stderr = FALSE
    ))
    if (identical(status, 0L)) {
      return(candidate)
    }
  }
  NULL
}

# Path to the EUR_subset bed file used in the OpenADMIXTURE.jl tests (379
# individuals, 54,051 SNPs), downloaded from the SnpArrays.jl repository the
# first time it is needed. Files are cached in OADM_TEST_DATA if set, or in
# the user cache directory of the package. Returns NULL if the files cannot
# be downloaded.
eur_subset_bed <- function() {
  data_dir <- Sys.getenv(
    "OADM_TEST_DATA",
    tools::R_user_dir("openadmixturer", which = "cache")
  )
  dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
  base_url <- "https://github.com/OpenMendel/SnpArrays.jl/raw/master/data/"
  for (ext in c("bed", "bim", "fam")) {
    dest <- file.path(data_dir, paste0("EUR_subset.", ext))
    if (!file.exists(dest)) {
      ok <- tryCatch(
        utils::download.file(paste0(base_url, "EUR_subset.", ext), dest,
          mode = "wb", quiet = TRUE
        ) == 0,
        error = function(e) FALSE,
        warning = function(w) FALSE
      )
      if (!ok) {
        unlink(dest)
        return(NULL)
      }
    }
  }
  file.path(data_dir, "EUR_subset.bed")
}

# Reorder the columns of q to best match q_ref (the labels of the clusters
# are arbitrary); returns the permutation
match_clusters_perm <- function(q, q_ref) {
  k <- ncol(q)
  perms <- function(n) {
    if (n == 1) {
      return(matrix(1))
    }
    sub <- perms(n - 1)
    do.call(rbind, lapply(seq_len(n), function(i) {
      cbind(i, matrix(setdiff(seq_len(n), i)[sub], nrow = nrow(sub)))
    }))
  }
  all_perms <- perms(k)
  diffs <- apply(all_perms, 1, function(x) sum(abs(q[, x] - q_ref)))
  all_perms[which.min(diffs), ]
}
