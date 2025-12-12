#' Predict method for tune.mff objects
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
#' @param object An object of class \code{tune.mff}.
#' @param pred_matrix Matrix of base model predictions (n x p).
#' @param type Character string specifying the prediction mode.
#'
#'   \code{"best"} returns predictions from the MFF model that achieved the
#'   optimal value according to the training metric (e.g., RMSE, MAE).
#'
#'   \code{"all"} returns a matrix or list of predictions from every MFF model
#'   generated during the tuning process.
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
#'  mff_tune_model <- tune.mff(result_train$pred_matrix_valid, result_train$y_valid, max_c = 6,
#'  iter.max=100,nstart = 100,mff.method = "kmeans")
#'  X <- result_train$pred_matrix_test
#'  out <- predict(mff_tune_model, X)
#'
#' head(out$mff_preds)   # cluster-based outputs
#' out$mff_weights       # rounded memberships
#'
#' }
#'
#' @export
predict.tune.mff <- function(object,
                             pred_matrix,
                             type = c("best", "all"),
                             ...) {

  type <- match.arg(type, c("best","all"))
  if (type == "best") {
    memberships <- object$best_weight
    preds <- pred_matrix %*% memberships
    return(list(
      mff_preds   = preds,
      mff_weights = round(memberships, 4)
    ))
  }

  memberships <- object$memberships
  preds <- pred_matrix %*% memberships
  return(list(
    mff_preds   = preds,
    mff_weights = round(memberships, 4)
  ))
}
