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
#' @param predicted Numeric vector or numeric matrix of predicted values.
#'        If a matrix is provided, each column is evaluated separately.
#'
#' @param actual Numeric vector of true target values corresponding to
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
#'   x <- seq(100)
#'   y <- 2 * x + rnorm(100)
#'   model <- lm(y ~ x)
#'   pred <- predict(model)
#'   evaluate(pred,y)
#' }
#'
#' @importFrom stats median
#'
#' @export
evaluate <- function(predicted, actual) {

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

  if (!is.matrix(predicted)) {
    m <- compute_metrics(predicted, actual)
    return(t(m))
  }

  results <- t(apply(predicted, 2, function(p) compute_metrics(p, actual)))

  return(results)
}
