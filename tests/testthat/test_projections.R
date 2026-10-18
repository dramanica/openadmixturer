# reference Euclidean projection onto {sum(q) == 1, q >= eps}, by bisection
# on the threshold t solving sum(pmax(y - t, eps)) == 1
project_q_reference <- function(y, eps = 1e-5) {
  f <- function(t) sum(pmax(y - t, eps)) - 1
  lo <- min(y) - 1
  hi <- max(y)
  for (i in 1:200) {
    mid <- (lo + hi) / 2
    if (f(mid) > 0) lo <- mid else hi <- mid
  }
  pmax(y - (lo + hi) / 2, eps)
}

test_that("project_q lands on the shifted simplex", {
  set.seed(3)
  for (k in c(2, 3, 5, 10, 20, 70)) {
    y <- matrix(rnorm(k * 50, sd = 2), k, 50)
    y[, 1:10] <- runif(k * 10, -0.5, 1.5)
    q <- oadm_project_q_cpp(y)
    expect_equal(colSums(q), rep(1, 50), tolerance = 1e-12)
    expect_true(all(q >= 1e-5 - 1e-15))
    expect_equal(q, apply(y, 2, project_q_reference), tolerance = 1e-10)
  }
})

test_that("project_q is idempotent and leaves its input alone", {
  set.seed(4)
  y <- matrix(rexp(40), 4, 10)
  y_copy <- y + 0
  q <- oadm_project_q_cpp(y)
  expect_identical(y, y_copy)
  expect_equal(oadm_project_q_cpp(q), q, tolerance = 1e-14)
  feasible <- matrix(c(0.2, 0.3, 0.5, 1e-5, 0.5, 0.5 - 1e-5), 3, 2)
  expect_equal(oadm_project_q_cpp(feasible), feasible, tolerance = 1e-14)
})

test_that("project_p clamps to [eps, 1 - eps]", {
  p <- matrix(c(-1, 0, 1e-6, 0.3, 1, 2), 2, 3)
  expect_identical(
    oadm_project_p_cpp(p),
    matrix(c(1e-5, 1e-5, 1e-5, 0.3, 1 - 1e-5, 1 - 1e-5), 2, 3)
  )
  expect_identical(p[1, 1], -1)
})
