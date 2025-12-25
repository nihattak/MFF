#' Predict method for mff objects
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
#' @param object An object of class \code{mff}.
#' @param pred_matrix Matrix of base model predictions.
#' @param type Character string specifying the prediction mode. Option for MFF tune function \code{tune.mff()}.
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
#'  boston <- MASS::Boston
#'  result_train <- model.train(
#'     target = "medv",
#'     data = boston,
#'     ntest = 50,
#'     nvalid = 50
#'  )
#'
#'  mff_tune_model <- tune.mff(result_train$pred_matrix_valid, result_train$y_valid, max_c = 6,
#'  iter.max=100,nstart = 100,mff.method = "kmeans")
#'  X <- result_train$pred_matrix_test
#'  out <- predict(mff_tune_model, X)
#'
#' head(out$mff_preds)
#' out$mff_weights
#'
#' }
#'
#' @export
predict.mff <- function(object,
                             pred_matrix,
                             type = c("best", "all"),
                             ...) {

  type <- match.arg(type)

  if (type == "best") {

    weights <- if (!is.null(object$best_weight)) {
      object$best_weight
    } else {
      object$weights
    }

  } else {
    weights <- object$weights
  }

  preds <- pred_matrix %*% weights

  list(
    mff_preds   = preds,
    mff_weights = round(weights, 4)
  )
}

