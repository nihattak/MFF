#' Generate Membership-Weighted Cluster Predictions
#'
#' This function produces cluster-specific predicted values by combining the
#' model prediction matrix with a fuzzy membership matrix. Each resulting
#' column represents a separate cluster-based output obtained through
#' membership-weighted aggregation of model predictions.
#'
#' @description
#' The function performs a weighted linear combination:
#'
#' \deqn{ \hat{Y}_{cluster} = X \times U }
#'
#' where:
#' \itemize{
#'   \item \eqn{X}: model prediction matrix (\code{samples × models})
#'   \item \eqn{U}: membership matrix (\code{models × clusters})
#' }
#'
#' Each cluster output reflects how strongly each model contributes according
#' to its membership degree. Membership matrices usually originate from
#' meta-fuzzy methods such as FCM, GK, FKMGK, PFCM, or MKF (K-means–based).
#'
#' @param pred_matrix A numeric matrix containing predictions from multiple
#'        models. Rows correspond to observations and columns correspond
#'        to the individual base models.
#'
#' @param memberships A numeric matrix of membership degrees with dimensions
#'        \code{models × clusters}. The number of rows must match the number
#'        of columns in \code{pred_matrix}.
#'
#' @param prefix Character prefix used for naming the cluster-based output
#'        columns. Default is \code{"Cluster_"}.
#'
#' @details
#' Membership values are not normalized inside this function; they should be
#' standardized prior to calling \code{predict()} when needed.
#'
#' The function also returns a rounded version of the membership matrix for
#' inspection, reporting, or visualization purposes.
#'
#' @return A list with:
#' \describe{
#'   \item{preds}{A matrix of cluster-specific predicted values
#'         (\code{samples × clusters}).}
#'   \item{weights}{The membership matrix rounded to four decimals.}
#' }
#'
#' @examples
#' \dontrun{
#' # X: model prediction matrix (samples × models)
#' # U: membership matrix (models × clusters)
#' out <- predict(X, U)
#'
#' head(out$preds)   # cluster-based outputs
#' out$weights       # rounded memberships
#' }
#'
#' @seealso
#' \code{mff()} for generating membership matrices using meta-fuzzy methods.
#'
#' @export
predict.mff <- function(object, pred_matrix, prefix="Cluster_", ...) {

  memberships <- object  # object = weight vector/matrix

  preds <- pred_matrix %*% memberships
  colnames(preds) <- paste0(prefix, 1:ncol(as.matrix(memberships)))

  return(list(
    preds = preds,
    weights = round(memberships, 4)
  ))
  # predict.mff <- function(pred_matrix, memberships,prefix = "Cluster_") {
  #   preds <- pred_matrix %*% memberships
  #   colnames(preds) <- paste0(prefix, 1:ncol(as.matrix(memberships)))
  #   return(list(preds = preds,weights = round(memberships,4)))
  # }
}

