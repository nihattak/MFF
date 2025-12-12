#' Generate Meta-Fuzzy Memberships and Cluster-Level Predictions
#'
#' This function computes model-level fuzzy or probabilistic memberships using
#' several meta-fuzzy clustering methods applied to the model prediction matrix \code{x}.
#' Each base model is represented by its prediction vector across samples, and fuzzy
#' memberships are estimated in this meta-prediction space. The resulting membership
#' matrix is then used to produce cluster-wise meta-fusion predictions via
#' \code{mff.predict()}, and cluster-level evaluation statistics are calculated.
#'
#' @description
#' \code{mff.gen()} serves as the core generator for the Meta-Fuzzy Fusion (MFF)
#' framework. It supports five membership-building methods:
#'
#' \itemize{
#'   \item \strong{"fcm"}: Classical Fuzzy C-Means (Euclidean, spherical clusters)
#'   \item \strong{"gk"}: Gustafson–Kessel Fuzzy Clustering (adaptive anisotropic shapes)
#'   \item \strong{"fkmgk"}: Hybrid Fuzzy K-Means with GK adaptation
#'   \item \strong{"pfcm"}: Probabilistic Fuzzy C-Means (softmax-like membership)
#'   \item \strong{"kmeans"}: Deterministic K-means converted to fuzzy memberships (MKF)
#' }
#'
#' All methods produce a membership matrix with dimensions
#' \code{models × clusters}, fully compatible with \code{mff.predict()}, which then
#' computes cluster-wise meta-predictions by linearly combining model predictions
#' weighted by fuzzy memberships.
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
#'        One of \code{"fcm"}, \code{"gk"}, \code{"fkmgk"}, \code{"pfcm"}, or \code{"kmeans"}.
#'
#' @details
#' The prediction matrix \code{x} is internally transposed so that each base model
#' is treated as an individual observation in the meta-clustering procedure.
#' Membership matrices are standardized column-wise using \code{prop.table()}
#' to ensure that the total membership weight within each cluster sums to 1.
#'
#' After membership estimation, the function applies \code{mff.predict()} to construct
#' cluster-specific meta-predictions using:
#'
#' \deqn{ \hat{Y}_{cluster} = X \times U }
#'
#' where:
#' \itemize{
#'   \item \eqn{X}: model prediction matrix (\code{samples × models})
#'   \item \eqn{U}: fuzzy membership matrix (\code{models × clusters})
#' }
#'
#' Cluster-wise performance metrics are computed using \code{evaluate()}.
#'
#' @return A list containing:
#' \describe{
#'   \item{method}{The clustering method used.}
#'   \item{standartized_membership}{Column-standardized membership matrix
#'         (\code{models × clusters}).}
#'   \item{cluster_preds}{Matrix of cluster predictions.}
#'   \item{cluster_scores}{Evaluation metrics for each cluster.}
#' }
#'
#' @seealso
#' \code{mff.predict()} for cluster-wise meta-fusion,
#' \code{cmeans()}, \code{gk()}, \code{FKM.gk()}, \code{pfcm()}, \code{kmeans()}
#' for membership generation algorithms.
#'
#' @examples
#' \dontrun{
#' # Generate fuzzy memberships and cluster predictions using Gustafson–Kessel
#' result <- mff(x = pred_matrix, y = y_true, c = 3, method = "gk")
#'
#' # Inspect membership weights
#' result$standartized_membership
#'
#' # Cluster-level meta predictions
#' result$cluster_preds
#'
#' # Performance per cluster
#' result$cluster_scores
#' }
#'
#' @importFrom e1071 cmeans
#' @importFrom ppclust gk pfcm
#' @importFrom fclust FKM.gk
#'
#' @export
mff <- function(x, y, c = 3, m = 2, eta = 2,iter.max=1000,nstart = 100,method = c("fcm", "gk","fkmgk", "pfcm", "kmeans")) {
  method <- match.arg(method)

  if (method == "fcm") {
    result <- cmeans(t(x), centers = c, m = m,iter.max=iter.max)
    membership <- result$membership

  } else if (method == "gk") {
    result <- gk(t(x), centers = c, m = m,stand = T,iter.max=iter.max)
    membership <- result$u

  } else if (method == "fkmgk") {
    result <- FKM.gk(t(x), k = c, m = m,RS = 100,stand = T,maxit=iter.max)
    membership <- result$U

  }else if (method == "pfcm") {
    result <- pfcm(t(x), centers = c, m = m, eta = eta,stand = T,iter.max=iter.max)
    membership <- result$u

  } else if (method == "kmeans") {
    result <- kmeans(t(x), centers = c, nstart = nstart, iter.max = iter.max)
    membership <- mkf_weight(result$cluster)
  } else {
    stop("Unknown method.")
  }

  standartized_membership <- prop.table(membership, margin = 2)
  rownames(standartized_membership) <- colnames(x)

  standartized_membership <- structure(
    standartized_membership,
    class = c("mff", class(standartized_membership))
  )

  cluster_preds <- predict(standartized_membership,x)
  cluster_scores <- evaluate(cluster_preds$preds, y)

  out <- list(
    method = method,
    standartized_membership = standartized_membership,
    cluster_preds = cluster_preds$preds,
    cluster_scores = cluster_scores
  )

  return(out)
}
