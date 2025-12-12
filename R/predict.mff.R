#' Generate Membership-Weighted Cluster Predictions
#'
#' This function produces cluster-specific predicted values by combining the
#' model prediction matrix with a fuzzy membership matrix. Each resulting
#' column represents a separate cluster-based output obtained through
#' membership-weighted aggregation of model predictions.
#'
#' @description
#' The function performs a crisp/fuzzy weighted linear combination.
#'
#' \deqn{\mathrm{MFF}_{cluster} = w \times X}
#'
#' where:
#' \itemize{
#'   \item \eqn{w}: weight matrix calculated by crisp/fuzzy memberships
#'   \item \eqn{X}: model prediction matrix
#' }
#'
#' Each cluster output reflects how strongly each model contributes according
#' to its membership degree. Membership matrices usually originate from
#' meta-fuzzy methods such as FCM, GK,or PFCM.
#'
#' @param object A trained model (MFF).
#'
#' @param pred_matrix A numeric matrix containing predictions from multiple
#'        models. Rows correspond to observations and columns correspond
#'        to the individual base models.
#' @param cluster A number to select specific cluster to predict.
#' @param ... Not used.
#'
#' @details
#'
#' P.S. The function also returns a rounded version of the membership matrix for
#' inspection, reporting, or visualization purposes.
#'
#' @return A list with:
#' \describe{
#'   \item{mff_preds}{A matrix of cluster-specific predicted values.}
#'   \item{mff_weights}{The weight matrix rounded to four decimals if cluster number is not specified
#'   If the number of cluster is specified it returns a weight vector of the selected cluster.}
#' }
#'
#' @examples
#' \dontrun{
#' # X: model prediction matrix
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
#'  X <- result_train$pred_matrix_test
#'  out <- predict(mff_model, X)
#'
#' head(out$mff_preds)   # cluster-based outputs
#' out$mff_weights       # rounded weights
#'
#' }
#'
#' @seealso
#' \code{mff()} for generating membership matrices using meta-fuzzy methods.
#'
#' @noRd
#' @export
predict.mff <- function(object,
                        pred_matrix,
                        cluster = NULL,
                        ...) {

  memberships <- object$weight_matrix
  preds <- pred_matrix %*% memberships
  k <- ncol(preds)

  # --- No cluster specified -> return ALL ---------------------------
  if (is.null(cluster)) {
    return(list(
      mff_preds = preds,
      mff_weights = round(memberships, 4)
    ))
  }

  # --- Validation ---------------------------------------------------
  if (!is.numeric(cluster) || any(cluster %% 1 != 0)) {
    stop("`cluster` must be integer index(es).")
  }

  if (any(cluster < 1 | cluster > k)) {
    stop("`cluster` index out of bounds. Must be between 1 and ", k, ".")
  }

  # --- Return selected cluster(s) -----------------------------------
  selected_preds <- preds[, cluster, drop = FALSE]
  selected_weights <- memberships[, cluster, drop = FALSE]

  return(list(
    mff_preds = selected_preds,
    mff_weights = round(selected_weights, 4)
  ))
}

