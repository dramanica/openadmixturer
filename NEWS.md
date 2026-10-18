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
