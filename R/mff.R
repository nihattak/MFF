#' Generate Meta-Fuzzy Memberships and Cluster-Level Predictions
#'
#' This function computes model-level fuzzy or probabilistic memberships using
#' several meta-fuzzy clustering methods applied to the model prediction matrix \code{x}.
#' Each base model is represented by its prediction vector across samples, and fuzzy
#' memberships are estimated in this meta-prediction space. The resulting membership
#' matrix is then used to produce cluster-wise predictions via
#' \code{predict()}, and cluster-level evaluation statistics are calculated.
#'
#' @description
#' \code{mff()} serves as the core generator for the Meta-Fuzzy Function (MFF)
#' framework. It supports four membership-building methods:
#'
#' \itemize{
#'   \item \strong{"fcm"}: Classical Fuzzy C-Means (Euclidean, spherical clusters)
#'   \item \strong{"gk"}: Gustafson–Kessel Fuzzy Clustering (adaptive anisotropic shapes)
#'   \item \strong{"pfcm"}: Probabilistic Fuzzy C-Means (softmax-like membership)
#'   \item \strong{"kmeans"}: Deterministic K-means converted to fuzzy memberships
#' }
#'
#' All methods produce a weight matrix with dimensions
#' \code{models × clusters}, fully compatible with \code{predict()}, which then
#' computes cluster-wise meta-predictions by linearly combining model predictions
#' weighted by crisp/fuzzy memberships.
#'
#' @param x A numeric matrix of model predictions with dimensions
#'        \code{samples × models}. Each column represents a base learner.
#'
#' @param y A numeric vector containing the true response values corresponding
#'        to the rows of \code{x}. Used for evaluating cluster-wise meta-predictions.
#'
#' @param c Integer. Number of fuzzy clusters/components to estimate.
#'
#' @param m Numeric. Fuzzy exponent parameter (usually \code{m = 2}). Controls
#'        the degree of fuzziness in FCM-type methods.
#'
#' @param eta Numeric. Regularization parameter for the probabilistic FCM
#'        (\code{"pfcm"}) method. Controls the sharpness of probabilistic weights.
#'
#' @param method Character string specifying the membership generation method.
#'        One of \code{"fcm"}, \code{"gk"}, \code{"pfcm"}, or \code{"kmeans"}.
#'
#' @param iter.max Integer. The maximum number of iterations allowed.
#'
#' @param nstart Integer. K-means only. If \code{centers} is a number, how many random sets should be chosen.
#'
#' @details
#' The prediction matrix \code{x} is internally transposed so that each base model
#' is treated as an individual observation in the meta-clustering procedure.
#' Membership matrices are standardized column-wise using \code{prop.table()}
#' to ensure that the total membership weight within each cluster sums to 1.
#'
#' After membership estimation, the function applies \code{predict()} to construct
#' cluster-specific meta-predictions using:
#'
#' \deqn{\mathrm{MFF}_{cluster} = w \times X}
#'
#' where:
#' \itemize{
#'   \item \eqn{w}: weight matrix calculated by crisp/fuzzy memberships
#'   \item \eqn{X}: model prediction matrix
#' }
#'
#' Cluster-wise performance metrics are computed using \code{evaluate()}.
#'
#' @return A list containing:
#' \describe{
#'   \item{method}{The clustering method used.}
#'   \item{weights}{Column-standardized membership matrix.}
#'   \item{cluster_preds}{Matrix of cluster predictions.}
#'   \item{cluster_scores}{Evaluation metrics for each cluster.}
#' }
#'
#' @seealso
#' \code{predict()} for cluster-wise prediction,
#' \code{cmeans()}, \code{gk()}, \code{pfcm()}, \code{kmeans()}
#' for membership generation algorithms.
#'
#' @examples
#' \dontrun{
#'  boston <- MASS::Boston
#'  result_train <- model.train(
#'     target = "medv",
#'     df = boston,
#'     test_count = 50,
#'     valid_count = 50
#'  )
#'
#'  mff_model <- mff(result_train$pred_matrix_valid, result_train$y_valid, c = 4,
#'  iter.max=100,nstart = 100,method = "kmeans")
#'  mff_model
#'}
#' @importFrom e1071 cmeans
#' @importFrom ppclust gk pfcm
#' @importFrom stats kmeans
#'
#' @export
mff <- function(x, y, c = 3, m = 2, eta = 2,iter.max=1000,nstart = 100,method = c("fcm", "gk", "pfcm", "kmeans")) {
  if (c > ncol(x)) {
    stop(sprintf(
      "Number of clusters (%d) cannot exceed the number of models (%d).",
      c, ncol(x)
    ))
  }

  method <- match.arg(method)

  if (method == "fcm") {
    result <- cmeans(t(x), centers = c, m = m,iter.max=iter.max)
    membership <- result$membership

  } else if (method == "gk") {
    result <- gk(t(x), centers = c, m = m,stand = T,iter.max=iter.max)
    membership <- result$u

  } else if (method == "pfcm") {
    result <- pfcm(t(x), centers = c, m = m, eta = eta,stand = T,iter.max=iter.max)
    membership <- result$u

  } else if (method == "kmeans") {
    result <- kmeans(t(x), centers = c, nstart = nstart, iter.max = iter.max)
    membership <- mkf_weight(result$cluster)
  } else {
    stop("Unknown method.")
  }

  weight_matrix <- prop.table(membership, margin = 2)
  rownames(weight_matrix) <- colnames(x)

  cluster_preds <- x %*% weight_matrix
  cluster_scores <- evaluate(cluster_preds, y)

  out <- list(
    method = method,
    weights = weight_matrix,
    cluster_scores = cluster_scores
  )

  out <- structure(out, class = "mff")

  return(out)
}
