#' Hyperparameter Search for Meta-Fuzzy Function
#'
#' This function performs a structured search over hyperparameters used to obtain
#' the best meta fuzzy functions. For a given model prediction matrix and
#' validation target vector, it evaluates multiple configurations of
#' \code{m}, \code{c}, and \code{eta} depending on the selected clustering
#' method. The function returns the best-performing configuration according
#' to the specified evaluation metric.
#'
#' @description
#' \code{tune.mff()} iterates over a predefined grid of parameter combinations
#' to identify the configuration that yields the lowest validation error.
#' Weights are generated using \code{mff()}, and cluster-based predicted
#' values are evaluated using the provided metric (\code{MAE}, \code{RMSE},
#' \code{MAPE},\code{SMAPE},\code{MSE} or \code{MedAE}).
#'
#' The search spans different parameter spaces depending on the selected method:
#' \itemize{
#'   \item \strong{FCM}: searches over \code{m} and \code{c}.
#'   \item \strong{GK}: searches over \code{m} and \code{c} with coarser steps.
#'   \item \strong{PFCM}: searches over \code{m}, \code{c}, and \code{eta}.
#'   \item \strong{K-Means}: searches only over \code{c}.
#' }
#'
#' @param x A numeric matrix containing model predictions, where rows
#'        represent observations and columns represent base models.
#'
#' @param y A numeric vector of true validation values used to evaluate
#'        cluster-based predicted outputs.
#'
#' @param max_c Integer. Maximum number of clusters to search up to.
#'
#' @param m_seq Numeric vector. Search values for the fuzziness index parameter \code{m}.
#'
#' @param eta_seq Numeric vector. Search values for the possibilistic parameter
#'        \code{eta} searched when \code{mff.method = "pfcm"}.
#'
#' @param iter.max Integer. The maximum number of iterations allowed.
#'
#' @param nstart Integer. K-means only. If \code{centers} is a number, how many random sets should be chosen.
#'
#' @param seed Integer. Seed for reproducibility during weight computation.
#'
#' @param mff.method Character string specifying the meta-fuzzy functions.
#'        One of \code{"fcm"}, \code{"gk"}, \code{"pfcm"},
#'        or \code{"kmeans"}.
#'
#' @param eval.method Character string specifying the metric used to select the
#'        best configuration. One of \code{MAE}, \code{RMSE},
#' \code{MAPE},\code{SMAPE},\code{MSE} or \code{MedAE}.
#'
#' @param logging Logical. If \code{TRUE}, prints progress information during the
#'        search.
#'
#' @details
#' For each parameter combination in the search grid, \code{mff()} is
#' executed to compute weights, cluster-based predictions, and evaluation
#' statistics. The best configuration is determined by the minimum value among
#' the cluster-specific scores for the selected evaluation metric.
#'
#' The output also includes:
#' \itemize{
#'   \item The best cluster index (cluster yielding minimum error)
#'   \item The weights corresponding to the best cluster
#'   \item The set of hyperparameters that achieved the best score
#' }
#'
#' @return A list containing:
#' \describe{
#'   \item{algorithm}{Selected clustering method.}
#'   \item{eval.method}{Evaluation metric used in the search.}
#'   \item{weights}{Weight matrix associated with the best configuration.}
#'   \item{best_params}{List of hyperparameters that obtained the best score.}
#'   \item{best_cluster}{Index of the cluster giving the lowest error.}
#'   \item{best_weight}{Weight vector for the best-performing cluster.}
#'   \item{best_scores}{Full evaluation scores for all clusters under the best configuration.}
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
#'  cluster_res <- tune.mff(result_train$pred_matrix_valid, result_train$y_valid,
#'  max_c = 6, iter.max=100,nstart = 100,mff.method = "kmeans")
#'
#'  cluster_res
#' }
#'
#' @seealso
#' \code{mff} for generating weight and cluster-based predictions.
#'
#' @export
tune.mff <- function(x, y, max_c = 5, m_seq = seq(1.1, 3, by = 0.1),eta_seq = seq(1.1, 3, by = 0.4), iter.max = 1000, nstart = 100,
                     seed = 123, mff.method = c("fcm", "gk","pfcm", "kmeans") , eval.method = c("MAE","RMSE","MAPE","SMAPE","MSE","MedAE"),logging = T) {
  if (max_c > ncol(x)) {
    stop(sprintf(
      "Number of clusters (%d) cannot exceed the number of models (%d).",
      max_c, ncol(x)
    ))
  }

  best_method <- Inf
  best_c <- NA
  best_m <- NA
  best_eta <- NA

  mff.method <- match.arg(mff.method)
  eval.method <- match.arg(eval.method)

  if (mff.method == "pfcm") {
    search_grid <- expand.grid(
      m = m_seq,
      c = 2:max_c,
      eta = eta_seq,
      iter.max = iter.max,
      nstart = nstart,
      KEEP.OUT.ATTRS = FALSE
    )
  } else if (mff.method %in% c("fcm")) {
    search_grid <- expand.grid(
      m = m_seq,
      c = 2:max_c,
      iter.max = iter.max,
      KEEP.OUT.ATTRS = FALSE
    )
  } else if (mff.method %in% c("gk")) {
    search_grid <- expand.grid(
      m = m_seq,
      c = 2:max_c,
      iter.max = iter.max,
      nstart = nstart,
      KEEP.OUT.ATTRS = FALSE
    )
  } else if (mff.method == "kmeans") {
    search_grid <- expand.grid(c = 2:max_c,nstart = nstart,iter.max=iter.max, KEEP.OUT.ATTRS = FALSE)
  } else {
    stop("Unknown method.")
  }

  if(logging) cat("Number of Combinations:",nrow(search_grid),"\nIterations: ")

  for (i in 1:nrow(search_grid)) {
    set.seed(seed)
    params <- as.list(search_grid[i, , drop = FALSE])

    mff_result <- mff(
      x = x,
      y = y,
      c = params$c,
      m = params$m,
      eta = params$eta,
      iter.max = params$iter.max,
      nstart = params$nstart,
      method = mff.method
    )

    if(logging){
      cat(i,"",sep = " ")
    }

    current_metric <- min(mff_result$cluster_scores[,eval.method])

    if (current_metric < best_method) {
      best_preds <- mff_result$cluster_preds
      best_method <- current_metric
      best_params <- params
      weights <- mff_result$weights
      best_scores <- mff_result$cluster_scores
    }
  }

  if(logging) cat("\n")

  idx <- unname(which.min(best_scores[,eval.method]))

  best_weight <- weights[, idx]

  out <- list(
    algorithm = mff.method,
    eval.method = eval.method,
    weights = weights,
    best_params = best_params,
    best_cluster = idx,
    best_weight = best_weight,
    best_scores = best_scores
  )

  out <- structure(
    out,
    class = "mff"
  )

  return(out)
}
