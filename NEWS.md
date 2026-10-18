# openadmixturer 0.0.0.9000

* Initial development version: `gt_openadmixture()`, a CPU (OpenMP)
  reimplementation of OpenADMIXTURE.jl that reads genotypes directly from
  `gen_tibble` backing files.
* Numerical safeguards that deviate from OpenADMIXTURE.jl: guard against
  vanishing pivots in the active-set QP, discard non-finite Newton steps,
  and make the projections robust to non-finite values. Also avoid an
  integer overflow when clamping very large P matrices.
* Cross-validation as in ADMIXTURE (`crossval = TRUE`, `n_folds = 5`), filling
  the `cv` element of `gt_admix` objects, so that `autoplot(type = "cv")` can
  be used to choose K.
* Better multithreading with few individuals: the Q-side kernels also split
  the loci into segments when there are too few blocks of individuals to
  keep the threads busy (e.g. 50 individuals: 1.0x -> 3.5x speed-up of the
  Q update on 4 cores). Results remain identical for any number of cores.
* Lower memory use with many loci: the whole fitting loop now runs in C++
  (src/fit.cpp), with the estimates in buffers allocated once and reused at
  every iteration, and the quasi-Newton step computed directly on the Q and
  P blocks. Previously each iteration created new R matrices, and discarded
  copies of P accumulated until R's garbage collector ran. On 60
  individuals x 1.5 million loci (K = 4), peak memory fell from about 34 to
  9 copies of the parameters. Results are unchanged.
* Faster fits: the log-likelihood is only computed when the algorithm needs
  it, skipping one of the six passes over the genotypes in most iterations
  and the passes after each EM step (which are still reported every 5 steps
  with `verbose = TRUE`). Estimates are unchanged.
