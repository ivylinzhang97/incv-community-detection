test_that("edge.index.map maps u=1 to (1,2)", {
  res <- edge.index.map(1)
  expect_equal(res$x, 1)
  expect_equal(res$y, 2)
})

test_that("edge.index.map maps u=6 to (3,4)", {
  res <- edge.index.map(6)
  expect_equal(res$x, 3)
  expect_equal(res$y, 4)
})

test_that("edge.index.map handles vectorised input", {
  res <- edge.index.map(1:10)
  expect_length(res$x, 10)
  expect_length(res$y, 10)
})

test_that("neglog(1, 0.5) equals -log(0.5)", {
  expect_equal(neglog(1, 0.5), -log(0.5))
})

test_that("neglog(n, 0) returns 0", {
  expect_equal(neglog(5, 0), 0)
  expect_equal(neglog(100, 0), 0)
})

test_that("l2 of identical matrices is 0", {
  m <- matrix(runif(20), 4, 5)
  expect_equal(l2(m, m), 0)
})

test_that("bin.dev returns finite for valid inputs", {
  x <- c(0, 1, 0, 1)
  y <- c(0.1, 0.9, 0.2, 0.8)
  expect_true(is.finite(bin.dev(x, y)))
})

test_that("bin.dev works with matrix inputs", {
  x <- matrix(c(0, 1, 0, 1), nrow = 2)
  y <- matrix(c(0.1, 0.9, 0.2, 0.8), nrow = 2)
  expect_true(is.finite(bin.dev(x, y)))
})
