test_that("fast.SBM.est returns K x K matrix", {
  set.seed(42)
  net <- community.sim(k = 3, n = 120, n1 = 40, p = 0.5, q = 0.05)
  cl <- SBM.spectral.clustering(net$adjacency, k = 3)$cluster
  B <- fast.SBM.est(net$adjacency, cl, K = 3)
  expect_equal(dim(B), c(3, 3))
  expect_true(all(is.finite(B)))
})

test_that("best.perm.label.match returns identity for same labels", {
  lab <- c(1, 1, 2, 2, 3, 3)
  E <- best.perm.label.match(lab, lab)
  expect_equal(E, diag(3))
})

test_that("best.perm.label.match handles 2-community swap", {
  lab   <- c(2, 2, 2, 1, 1, 1)
  fixed <- c(1, 1, 1, 2, 2, 2)
  E <- best.perm.label.match(lab, fixed)
  expect_equal(sum(E), 2)
})

test_that("matched.lab aligns permuted labels", {
  lab   <- c(2, 2, 1, 1, 3, 3)
  fixed <- c(1, 1, 2, 2, 3, 3)
  result <- matched.lab(lab, fixed)
  expect_equal(result, fixed)
})
