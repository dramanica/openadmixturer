test_genotypes <- rbind(
  c(1, 1, 0, 1, 1, 0),
  c(2, 2, 0, NA, 0, 0),
  c(2, NA, 0, 0, 1, 1),
  c(2, 0, 0, 2, 0, 0),
  c(1, 2, 0, 1, 2, 1),
  c(0, 0, 0, 0, NA, 2),
  c(0, 1, 2, 0, 1, NA)
)

test_that("decoding matches show_genotypes on full data", {
  test_gt <- make_test_gt(test_genotypes)
  dat <- oadm_prepare(test_gt)
  expect_equal(dat$n_indiv, 7)
  expect_equal(dat$n_loci, 6)
  expect_equal(oadm_decode(dat), test_genotypes, ignore_attr = TRUE)
})

test_that("decoding respects unsorted row and column subsets", {
  test_gt <- make_test_gt(test_genotypes)
  sub_gt <- test_gt[c(5, 2, 7, 1), ]
  sub_gt <- tidypopgen::select_loci(sub_gt, c(6, 1, 4))
  dat <- oadm_prepare(sub_gt)
  expect_equal(
    oadm_decode(dat),
    test_genotypes[c(5, 2, 7, 1), c(6, 1, 4)],
    ignore_attr = TRUE
  )
  expect_equal(
    oadm_decode(dat),
    tidypopgen::show_genotypes(sub_gt),
    ignore_attr = TRUE
  )
})

test_that("pseudohaploids are rescaled to a single allele", {
  hap_genotypes <- test_genotypes
  # make individuals 2 and 4 fully homozygous
  hap_genotypes[2, ] <- c(2, 2, 0, NA, 0, 0)
  hap_genotypes[4, ] <- c(2, 0, 0, 2, 0, 0)
  test_gt <- make_test_gt(hap_genotypes)
  pseudo_gt <- tidypopgen::gt_pseudohaploid(test_gt)
  dat <- oadm_prepare(pseudo_gt)
  ploidy <- tidypopgen::indiv_ploidy(pseudo_gt)
  expect_equal(dat$n_trials, as.numeric(ploidy))
  expected <- hap_genotypes
  expected[ploidy == 1, ] <- expected[ploidy == 1, ] / 2
  expect_equal(oadm_decode(dat), expected, ignore_attr = TRUE)
})

test_that("non-diploid data are rejected", {
  test_gt <- make_test_gt(test_genotypes)
  attr(test_gt$genotypes, "ploidy") <- 4
  expect_error(oadm_prepare(test_gt), "diploid or pseudohaploid")
  expect_error(oadm_prepare(data.frame(a = 1)), "gen_tibble")
})
