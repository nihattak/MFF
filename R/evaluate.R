#' Compute Error Metrics for Predicted Values
#'
#' Computes multiple regression error metrics for predicted values by comparing
#' them to the true target values.
#'
#' @description
#' \code{evaluate()} supports two input formats:
#' \itemize{
#'   \item A numeric vector of predicted values.
#'   \item A numeric matrix where each column represents the predictions from
#'         a different model or cluster.
#' }
#'
#' For vector input, the function returns a single-row data frame.
#' For matrix input, metrics are computed column-wise and returned as a
#' multi-row data frame.
#'
#' @param y_pred Numeric vector or numeric matrix of predicted values.
#'        If a matrix is provided, each column is evaluated separately.
#'
#' @param y_true Numeric vector of true target values corresponding to
#'        the rows of \code{preds}.
#'
#' @details
#' The following performance metrics are computed:
#'
#' \strong{Mean Absolute Error (MAE):}
#' \deqn{ MAE = mean(|\hat{y} - y|) }
#'
#' \strong{Root Mean Square Error (RMSE):}
#' \deqn{ RMSE = \sqrt{mean((\hat{y} - y)^2)} }
#'
#' \strong{Mean Absolute Percentage Error (MAPE):}
#' \deqn{ MAPE = mean(|(\hat{y} - y)/(y + \varepsilon)|) \times 100 }
#'
#' \strong{Symmetric Mean Absolute Percentage Error (SMAPE):}
#' \deqn{ SMAPE = mean\left(\frac{2|\hat{y} - y|}{|\hat{y}| + |y| + \varepsilon}\right) \times 100 }
#'
#' \strong{Mean Squared Error (MSE):}
#' \deqn{ MSE = mean((\hat{y} - y)^2) }
#'
#' \strong{Median Absolute Error (MedAE):}
#' \deqn{ MedAE = median(|\hat{y} - y|) }
#'
#' A small constant \eqn{\varepsilon = 1e-8} is added to avoid division by zero
#' for percentage-based metrics.
#'
#' @return A data frame with the following columns:
#' \describe{
#'   \item{MAE}{Mean Absolute Error}
#'   \item{RMSE}{Root Mean Square Error}
#'   \item{MAPE}{Mean Absolute Percentage Error}
#'   \item{SMAPE}{Symmetric Mean Absolute Percentage Error}
#'   \item{MSE}{Mean Squared Error}
#'   \item{MedAE}{Median Absolute Error}
#' }
#'
#' @examples
#' \dontrun{
#'   boston <- MASS::Boston
#'   results <- model.train(
#'     target = "medv",
#'     df = boston,
#'     test_count = 50,
#'     valid_count = 50
#'   )
#'   evaluate(results$pred_matrix_valid,results$y_valid)
#' }
#'
#' @importFrom stats median
#'
#' @export
evaluate <- function(y_pred, y_true) {

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

  if (!is.matrix(y_pred)) {
    m <- compute_metrics(y_pred, y_true)
    return(t(m))
  }

  results <- t(apply(y_pred, 2, function(p) compute_metrics(p, y_true)))

  return(results)
}
