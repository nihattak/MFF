#' Compute error metrics for predicted values or prediction matrices
#'
#' @description
#' Compute error metrics for predicted values or prediction matrices by comparing them with
#' the corresponding true response values.
#'
#' @param predicted numeric vector or numeric matrix of predictions. If matrix, columns are evaluated
#' separately.
#' @param actual numeric vector of true target values.
#'
#' @details
#' The evaluate function is used to quantify the predictive performance of meta fuzzy function
#' predictions by comparing them with the corresponding true response values. It supports both
#' vector- and matrix-valued prediction inputs, allowing performance assessment of a single
#' meta fuzzy function as well as simultaneous evaluation of multiple meta fuzzy functions.
#' When a prediction vector is provided, the function computes a single set of performance
#' metrics; when a prediction matrix is provided, metrics are computed separately for each meta
#' fuzzy function.
#'
#' @return A matrix with columns MAE, RMSE, MAPE, SMAPE, MSE, and MedAE, with one row
#' per evaluated prediction vector.
#'
#' @seealso
#' \code{\link{mff}} for generating predictions,
#' \code{\link{model.train}} for preparing base model prediction matrices,
#' \code{\link{tune.mff}} for hyperparameter tuning performance.
#'
#' @examples
#' x <- seq(100)
#' y <- 2*x + stats::rnorm(100)
#' m <- stats::lm(y ~ x)
#' pred <- stats::predict(m)
#' evaluate(pred, y)
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
