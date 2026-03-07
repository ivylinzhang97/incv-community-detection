test_that("nscv.f.fold selects K from candidates", {
  set.seed(100)
  net <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)
  result <- nscv.f.fold(net$adjacency, k.vec = 2:5, f = 5)
  expect_true(result$k.loss %in% 2:5)
  expect_true(result$k.mse %in% 2:5)
  expect_length(result$cv.loss, 4)
  expect_length(result$cv.mse, 4)
})

test_that("nscv.f.fold recovers K=3 for well-separated SBM", {
  set.seed(100)
  net <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)
  result <- nscv.f.fold(net$adjacency, k.vec = 2:5, f = 5)
  expect_equal(result$k.loss, 3)
})

test_that("nscv.f.fold works with loss method", {
  set.seed(100)
  net <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)
  result <- nscv.f.fold(net$adjacency, k.vec = 2:4, f = 5, method = "loss")
  expect_true(result$k.loss %in% 2:4)
})

test_that("nscv.random.split runs and returns valid result", {
  set.seed(100)
  net <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)
  result <- nscv.random.split(net$adjacency, k.vec = 2:4,
                              split = 0.66, ite = 10)
  expect_true(result$k.chosen %in% 2:4)
  expect_length(result$cv.loss, 3)
})
