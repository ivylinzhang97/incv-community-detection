#' Spectral clustering for a Stochastic Block Model
#'
#' Performs spectral clustering on an adjacency matrix by computing the top
#' \code{k} singular vectors and applying k-means++ via \code{ClusterR::KMeans_rcpp}.
#'
#' @param A Symmetric adjacency matrix (n x n, binary or weighted).
#' @param k Number of clusters (default 2).
#' @return A list with component \code{cluster}, an integer vector of length n.
#' @export
#' @examples
#' net <- community.sim(k = 3, n = 90, n1 = 30, p = 0.5, q = 0.1)
#' cl <- SBM.spectral.clustering(net$adjacency, k = 3)
#' table(cl$cluster, net$membership)
SBM.spectral.clustering <- function(A, k = 2) {
  svd.k <- RSpectra::svds(A, k = k)
  U_k <- svd.k$u

  kpp_result <- ClusterR::KMeans_rcpp(U_k, clusters = k,
    num_init = 25, max_iters = 100,
    initializer = "kmeans++", fuzzy = FALSE)
  list(cluster = kpp_result$clusters)
}

#' Estimate SBM connection probabilities and negative log-likelihood
#'
#' Given a clustering and adjacency matrix, estimates the block probability
#' matrix and computes the negative log-likelihood.
#'
#' @param cluster Integer vector of cluster labels.
#' @param k Number of clusters.
#' @param A Adjacency matrix corresponding to the nodes in \code{cluster}.
#' @param restricted Logical. If \code{TRUE}, uses a restricted SBM with a
#'   single within-community probability \code{p} and a single
#'   between-community probability \code{q}. If \code{FALSE}, estimates a
#'   separate probability for each pair of communities.
#' @return A list with:
#'   \item{p.matrix}{k x k estimated block probability matrix.}
#'   \item{negloglike}{Negative log-likelihood of the model.}
#' @export
SBM.prob <- function(cluster, k, A, restricted = TRUE) {
  p.matrix <- matrix(0, nrow = k, ncol = k)
  negloglike <- 0

  edge.vector <- c(A)[c(upper.tri(A))]
  one <- edge.index.map(which(edge.vector == 1))
  zero <- edge.index.map(which(edge.vector == 0))

  if (restricted) {
    within.connect <- sum(cluster[one$x] == cluster[one$y])
    within.disconnect <- sum(cluster[zero$x] == cluster[zero$y])
    between.connect <- length(one$x) - within.connect
    between.disconnect <- length(zero$x) - within.disconnect

    within.total <- within.connect + within.disconnect
    between.total <- between.connect + between.disconnect
    if (within.total == 0) p <- 0
    else p <- within.connect / within.total
    if (between.total == 0) q <- 0
    else q <- between.connect / between.total
    diag(p.matrix) <- p
    p.matrix[(lower.tri(p.matrix)) | (upper.tri(p.matrix))] <- q

    negloglike <- neglog(within.connect, p) + neglog(within.disconnect, 1 - p) +
      neglog(between.connect, q) + neglog(between.disconnect, 1 - q)
  } else {
    for (i in 1:k) {
      for (j in i:k) {
        if (i == j) {
          connect <- sum((cluster[one$x] == i) & (cluster[one$y] == i))
          disconnect <- sum((cluster[zero$x] == i) & (cluster[zero$y] == i))
          total <- connect + disconnect
          if (total == 0) p <- 0
          else p <- connect / total
          p.matrix[i, i] <- p
          negloglike <- negloglike + neglog(connect, p) + neglog(disconnect, 1 - p)
        } else {
          connect <- sum((cluster[one$x] == i) & (cluster[one$y] == j)) +
            sum((cluster[one$x] == j) & (cluster[one$y] == i))
          disconnect <- sum((cluster[zero$x] == i) & (cluster[zero$y] == j)) +
            sum((cluster[zero$x] == j) & (cluster[zero$y] == i))
          total <- connect + disconnect
          if (total == 0) q <- 0
          else q <- connect / total
          p.matrix[i, j] <- q
          p.matrix[j, i] <- q
          negloglike <- negloglike + neglog(connect, q) + neglog(disconnect, 1 - q)
        }
      }
    }
  }
  list(p.matrix = p.matrix, negloglike = negloglike)
}
