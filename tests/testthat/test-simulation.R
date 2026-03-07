test_that("community.sim generates valid 2-community SBM", {
  set.seed(1)
  net <- community.sim(k = 2, n = 50, n1 = 20, p = 0.5, q = 0.1)
  expect_type(net, "list")
  expect_named(net, c("membership", "adjacency"))
  expect_length(net$membership, 50)
  expect_equal(dim(net$adjacency), c(50, 50))
  expect_true(isSymmetric(net$adjacency))
  expect_true(all(diag(net$adjacency) == 0))
  expect_equal(length(unique(net$membership)), 2)
})

test_that("community.sim generates valid 4-community SBM", {
  set.seed(2)
  net <- community.sim(k = 4, n = 120, n1 = 20, p = 0.4, q = 0.05)
  expect_equal(length(unique(net$membership)), 4)
  expect_equal(nrow(net$adjacency), 120)
})

test_that("community.sim.sbm generates distance-decaying SBM", {
  set.seed(3)
  net <- community.sim.sbm(n = 60, n1 = 20, eta = 0.3, rho = 0.1, K = 3)
  expect_true(all(c("adjacency", "membership", "conn") %in% names(net)))
  expect_equal(nrow(net$adjacency), 60)
  expect_equal(dim(net$conn), c(3, 3))
  expect_true(isSymmetric(net$adjacency))
})
