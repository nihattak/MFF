#' Compute Error Metrics for Predicted Values
#'
#' This function calculates standard prediction error metrics for either a single
#' predicted vector or multiple prediction columns supplied as a matrix. The
#' metrics computed are Mean Absolute Error (MAE), Root Mean Square Error (RMSE),
#' and Mean Absolute Percentage Error (MAPE).
#'
#' @description
#' \code{evaluate()} supports two input formats:
#' \itemize{
#'   \item A numeric vector of predicted values
#'   \item A matrix of predicted values, where each column represents a model or
#'         cluster-based prediction
#' }
#'
#' For vector input, the function returns a single-row data frame.
#' For matrix input, metrics are computed column-wise and returned as a
#' multi-row data frame.
#'
#' @param preds A numeric vector or numeric matrix containing predicted values.
#'        If a matrix is provided, each column is evaluated separately.
#'
#' @param y_true A numeric vector of true target values corresponding to the rows
#'        of \code{preds}.
#'
#' @param model_name Character string specifying the model name when evaluating
#'        a single prediction vector. Ignored when \code{preds} is a matrix.
#'
#' @details
#' The function computes the following metrics:
#'
#' \strong{Mean Absolute Error (MAE):}
#' \deqn{ MAE = mean(|\hat{y} - y|) }
#'
#' \strong{Root Mean Square Error (RMSE):}
#' \deqn{ RMSE = \sqrt{mean((\hat{y} - y)^2)} }
#'
#' \strong{Mean Absolute Percentage Error (MAPE):}
#' \deqn{ MAPE = mean(|(\hat{y} - y) / (y + \varepsilon)|) \times 100 }
#'
#' A small constant \eqn{\varepsilon = 1e-8} is added to avoid division by zero.
#'
#' When \code{preds} is a matrix, missing or empty column names are automatically
#' replaced with default names such as \code{"Model_1"}, \code{"Model_2"}, etc.
#'
#' @return A data frame containing:
#' \describe{
#'   \item{Model}{Model or prediction label.}
#'   \item{MAE}{Mean Absolute Error.}
#'   \item{RMSE}{Root Mean Square Error.}
#'   \item{MAPE}{Mean Absolute Percentage Error.}
#' }
#'
#' @examples
#' \dontrun{
#' # Vector input
#' evaluate(preds = y_pred, y_true = y)
#'
#' # Matrix input
#' evaluate(preds = pred_matrix, y_true = y)
#' }
#'
#' @export
evaluate <- function(preds, y_true, model_name = "Individual") {

  compute_metrics <- function(p, y) {
    errors <- p - y

    mae   <- mean(abs(errors))
    rmse  <- sqrt(mean(errors^2))
    mape  <- mean(abs(errors / (y + 1e-8))) * 100
    smape <- mean( 2 * abs(p - y) / (abs(p) + abs(y) + 1e-8) ) * 100
    mse   <- mean(errors^2)
    medae <- median(abs(errors))

    return(c(
      MAE = mae,
      RMSE = rmse,
      MAPE = mape,
      SMAPE = smape,
      MSE = mse,
      MedAE = medae
    ))
  }

  if (!is.matrix(preds)) {
    m <- compute_metrics(preds, y_true)
    df <- data.frame(
      Model = model_name,
      t(m),
      row.names = NULL
    )
    return(df)
  }

  model_names <- colnames(preds)
  if (is.null(model_names)) {
    model_names <- paste0("Model_", seq_len(ncol(preds)))
  } else {
    empty <- (model_names == "" | is.na(model_names))
    if (any(empty)) {
      model_names[empty] <- paste0("Model_", which(empty))
    }
  }

  results <- t(apply(preds, 2, function(p) compute_metrics(p, y_true)))

  df <- data.frame(
    Model = model_names,
    results,
    row.names = NULL
  )

  return(df)
}
