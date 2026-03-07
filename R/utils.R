#' Map a linear edge index to row-column indices in the upper triangle
#'
#' Given a linear index \code{u} over the upper triangle of a symmetric matrix
#' (column-major order), returns the corresponding row (\code{x}) and column
#' (\code{y}) indices. The mapping is independent of matrix size.
#'
#' @param u Integer vector of linear edge indices (1-based).
#' @return A list with components \code{x} (row indices) and \code{y} (column indices).
#' @export
#' @examples
#' edge.index.map(1:6)
edge.index.map <- function(u) {
  v <- (sqrt(1 + 8 * u) - 1) / 2
  f <- floor(v)
  y <- ceiling(v) + 1
  x <- ifelse(u == f * (f + 1) / 2, y - 1, u - f * (f + 1) / 2)
  list(x = x, y = y)
}

#' Safe negative log-likelihood term
#'
#' Computes \code{-n * log(p)}, returning 0 when \code{p <= 0}.
#'
#' @param n Numeric count.
#' @param p Probability (between 0 and 1).
#' @return Numeric value of \code{-n * log(p)} or 0 if \code{p <= 0}.
#' @export
neglog <- function(n = 1, p = 0.5) {
  ifelse(p <= 0, 0, -n * log(p))
}

#' Fast sum of matrix or vector elements
#' @param X A matrix or vector.
#' @return The total sum of all elements.
#' @keywords internal
sumFast <- function(X) {
  if (is.vector(X)) return(sum(X))
  sum(rowSums(X))
}

#' Find the mode of a vector
#' @param x A vector.
#' @return The most frequent value.
#' @keywords internal
modal <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#' L2 loss between two matrices
#'
#' @param x,y Numeric matrices of the same dimension.
#' @return The Frobenius norm \code{sqrt(sum((x - y)^2))}.
#' @export
l2 <- function(x, y) {
  sqrt(sum(rowSums((x - y)^2)))
}

#' Binomial deviance loss
#'
#' @param x Observed binary matrix/vector.
#' @param y Predicted probability matrix/vector.
#' @return Total binomial deviance (non-finite terms set to 0).
#' @export
bin.dev <- function(x, y) {
  tmp <- -x * log(y) - (1 - x) * log(1 - y)
  tmp[!is.finite(tmp)] <- 0
  if (is.matrix(tmp)) sum(rowSums(tmp)) else sum(tmp)
}

#' AUC via Wilcoxon statistic
#' @param score Numeric prediction scores.
#' @param bool Logical/binary label vector.
#' @return AUC value.
#' @keywords internal
auroc <- function(score, bool) {
  n1 <- sum(!bool)
  n2 <- length(score) - n1
  U  <- sum(rank(score)[!bool]) - n1 * (n1 + 1) / 2
  1 - U / n1 / n2
}

#' Negative AUC for matrix predictions
#'
#' @param A Observed binary adjacency (matrix or vector).
#' @param P Predicted probability (matrix or vector).
#' @return Negative AUC value (for minimisation-based model selection).
#' @export
AUC <- function(A, P) {
  -Rfast::auc(as(A, "vector"), as(P, "vector"))
}
