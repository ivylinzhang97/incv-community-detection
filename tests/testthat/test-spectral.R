test_that("SBM.spectral.clustering returns correct cluster vector", {
  set.seed(42)
  net <- community.sim(k = 3, n = 120, n1 = 40, p = 0.5, q = 0.05)
  cl <- SBM.spectral.clustering(net$adjacency, k = 3)
  expect_length(cl$cluster, 120)
  expect_true(all(cl$cluster %in% 1:3))
})

test_that("SBM.spectral.clustering works for k=2", {
  set.seed(42)
  net <- community.sim(k = 2, n = 80, n1 = 40, p = 0.5, q = 0.05)
  cl <- SBM.spectral.clustering(net$adjacency, k = 2)
  expect_length(cl$cluster, 80)
  expect_true(all(cl$cluster %in% 1:2))
})

test_that("SBM.prob restricted mode returns valid output", {
  set.seed(42)
  net <- community.sim(k = 3, n = 120, n1 = 40, p = 0.5, q = 0.05)
  cl <- SBM.spectral.clustering(net$adjacency, k = 3)$cluster
  res <- SBM.prob(cl, 3, net$adjacency, restricted = TRUE)
  expect_equal(dim(res$p.matrix), c(3, 3))
  expect_true(is.finite(res$negloglike))
  expect_true(res$p.matrix[1, 1] > res$p.matrix[1, 2])
})

test_that("SBM.prob unrestricted mode returns valid output", {
  set.seed(42)
  net <- community.sim(k = 3, n = 120, n1 = 40, p = 0.5, q = 0.05)
  cl <- SBM.spectral.clustering(net$adjacency, k = 3)$cluster
  res <- SBM.prob(cl, 3, net$adjacency, restricted = FALSE)
  expect_equal(dim(res$p.matrix), c(3, 3))
  expect_true(is.finite(res$negloglike))
})
