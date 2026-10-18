# Install ADMIXTURE, used by the vignette to compare openadmixturer with the
# original program, and by data-raw/regenerate_admixture_fixtures.R to create
# the ADMIXTURE results stored for tests/testthat/test_admixture.R (the tests
# themselves do not need ADMIXTURE).
#
# ADMIXTURE is installed in the conda environment "ctidygenclust" created by
# tidygenclust (https://github.com/EvolEcolGroup/tidygenclust), which
# tidypopgen::gt_admixture() finds automatically. Alternatively, put the
# admixture binary on the PATH.
#
# Run from the root of the package:
#   Rscript data-raw/setup_admixture.R

if (!requireNamespace("tidygenclust", quietly = TRUE)) {
  install.packages("tidygenclust",
    repos = c("https://evolecolgroup.r-universe.dev", getOption("repos"))
  )
}
tidygenclust::tgc_tools_install()
