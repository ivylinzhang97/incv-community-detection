test_that("ECV.for.blockmodel returns model selection", {
  set.seed(100)
  net <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)
  ecv <- ECV.for.blockmodel(net$adjacency, max.K = 4, B = 2,
                            holdout.p = 0.1, dc.est = 2)
  expect_type(ecv$dev.model, "character")
  expect_type(ecv$l2.model, "character")
  expect_type(ecv$auc.model, "character")
  expect_length(ecv$l2, 4)
  expect_length(ecv$dc.l2, 4)
})

test_that("NCV.for.blockmodel returns model selection", {
  set.seed(100)
  net <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)
  ncv <- NCV.for.blockmodel(net$adjacency, max.K = 4, cv = 3)
  expect_type(ncv$dev.model, "character")
  expect_type(ncv$l2.model, "character")
  expect_length(ncv$dev, 4)
})
