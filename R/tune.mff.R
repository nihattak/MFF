#' Hyperparameter Search for Meta-Fuzzy Methods
#'
#' This function performs a structured search over hyperparameters used in
#' meta-fuzzy membership generation. For a given model prediction matrix and
#' validation target vector, it evaluates multiple configurations of
#' \code{m}, \code{c}, and \code{eta} depending on the selected meta-fuzzy
#' method. The function returns the best-performing configuration according
#' to the specified evaluation metric.
#'
#' @description
#' \code{mff.search()} iterates over a predefined grid of parameter combinations
#' to identify the configuration that yields the lowest validation error.
#' Memberships are generated using \code{mff.gen()}, and cluster-based predicted
#' values are evaluated using the provided metric (\code{MAE}, \code{RMSE}, or
#' \code{MAPE}).
#'
#' The search spans different parameter spaces depending on the selected method:
#' \itemize{
#'   \item \strong{FCM}: searches over \code{m} and \code{c}.
#'   \item \strong{GK / FKMGK}: searches over \code{m} and \code{c} with coarser steps.
#'   \item \strong{PFCM}: searches over \code{m}, \code{c}, and \code{eta}.
#'   \item \strong{K-means (MKF)}: searches only over \code{c}.
#' }
#'
#' @param pred_matrix A numeric matrix containing model predictions, where rows
#'        represent observations and columns represent base models.
#'
#' @param y_valid A numeric vector of true validation values used to evaluate
#'        cluster-based predicted outputs.
#'
#' @param max_c Integer. Maximum number of clusters to search up to.
#'        The search spans \code{c = 2:max_c}.
#'
#' @param max_m Numeric. Maximum value of the fuzzy exponent parameter \code{m}
#'        to explore during the search.
#'
#' @param max_eta Numeric. Maximum value of the probabilistic parameter
#'        \code{eta} searched when \code{mff.method = "pfcm"}.
#'
#' @param seed Integer. Seed for reproducibility during membership computation.
#'
#' @param mff.method Character string specifying the meta-fuzzy method.
#'        One of \code{"fcm"}, \code{"gk"}, \code{"fkmgk"}, \code{"pfcm"},
#'        or \code{"kmeans"}.
#'
#' @param eval.method Character string specifying the metric used to select the
#'        best configuration. One of \code{"MAE"}, \code{"RMSE"}, or \code{"MAPE"}.
#'
#' @param logging Logical. If \code{TRUE}, prints progress information during the
#'        search.
#'
#' @details
#' For each parameter combination in the search grid, \code{mff.gen()} is
#' executed to compute memberships, cluster-based predictions, and evaluation
#' statistics. The best configuration is determined by the minimum value among
#' the cluster-specific scores for the selected evaluation metric.
#'
#' The output also includes:
#' \itemize{
#'   \item The best cluster index (cluster yielding minimum error)
#'   \item The membership weights corresponding to the best cluster
#'   \item Rounded membership matrix for interpretability
#'   \item The set of hyperparameters that achieved the best score
#' }
#'
#' @return A list containing:
#' \describe{
#'   \item{algo}{Selected meta-fuzzy method.}
#'   \item{eval.method}{Evaluation metric used in the search.}
#'   \item{memberships}{Membership matrix associated with the best configuration.}
#'   \item{memberships_rounded}{Rounded membership matrix.}
#'   \item{best_method}{Best (minimum) metric value observed.}
#'   \item{best_params}{List of hyperparameters that obtained the best score.}
#'   \item{best_cluster}{Index of the cluster giving the lowest error.}
#'   \item{best_weight}{Membership vector for the best-performing cluster.}
#'   \item{best_scores}{Full evaluation scores for all clusters under the best configuration.}
#' }
#'
#' @examples
#' \dontrun{
#' result <- tune.mff(
#'   pred_matrix = X_valid,
#'   y_valid = y_valid,
#'   max_c = 5,
#'   max_m = 3,
#'   mff.method = "gk",
#'   eval.method = "RMSE"
#' )
#'
#' result$best_params       # best m and c
#' result$best_cluster      # best performing cluster
#' result$memberships       # membership matrix
#' }
#'
#' @seealso
#' \code{mff.gen()} for generating memberships and cluster-based predictions.
#'
#' @importFrom e1071 cmeans
#' @importFrom ppclust gk pfcm
#' @importFrom fclust FKM.gk
#' @export
tune.mff <- function(pred_matrix, y_valid, max_c = 5, max_m = seq(1.1, 3, by = 0.1),max_eta = seq(1.1, 3, by = 0.4), iter.max = 1000, nstart = 100,
                       seed = 123, mff.method = c("fcm", "gk","fkmgk","pfcm", "kmeans") , eval.method = c("MAE","RMSE","MAPE"),logging = T) { # eval.method yerine metric te kullanılabilir
  best_method <- Inf
  best_c <- NA
  best_m <- NA
  best_eta <- NA

  mff.method <- match.arg(mff.method)
  eval.method <- match.arg(eval.method)

  if (mff.method == "pfcm") {
    search_grid <- expand.grid(
      m = max_m,
      c = 2:max_c,
      eta = max_eta,
      iter.max = iter.max,
      KEEP.OUT.ATTRS = FALSE
    )
  } else if (mff.method %in% c("fcm")) {
    search_grid <- expand.grid(
      m = max_m,
      c = 2:max_c,
      iter.max = iter.max,
      KEEP.OUT.ATTRS = FALSE
    )
  } else if (mff.method %in% c("gk","fkmgk")) {
    search_grid <- expand.grid(
      m = max_m,
      c = 2:max_c,
      iter.max = iter.max,
      KEEP.OUT.ATTRS = FALSE
    )
  } else if (mff.method == "kmeans") {
    search_grid <- expand.grid(c = 2:max_c,nstart = nstart,iter.max=iter.max, KEEP.OUT.ATTRS = FALSE)
  } else {
    stop("Unknown method.")
  }

  if(logging) cat("Number of Combinations:",nrow(search_grid),"\n")

  # print("Number of Combinations:")
  # print(nrow(search_grid))

  for (i in 1:nrow(search_grid)) {
    set.seed(seed)
    params <- as.list(search_grid[i, , drop = FALSE])

    mff_result <- mff(
      x = pred_matrix,
      y = y_valid,
      c = params$c,
      m = params$m,
      eta = params$eta,
      iter.max = params$iter.max,
      nstart = params$nstart,
      method = mff.method
    )
    if(logging) cat(i,"-iter\n",sep = "")

    current_metric <- min(mff_result$cluster_scores[[eval.method]])

    if (current_metric < best_method) {
      best_preds <- mff_result$cluster_preds
      best_method <- current_metric
      best_params <- params
      best_membership <- mff_result$standartized_membership
      best_scores <- mff_result$cluster_scores
    }
  }

  best_weight <- best_membership[, which.min(best_scores[[eval.method]])]

  best_weight <- structure(
    best_weight,
    class = c("mff", class(best_weight))
  )

  out <- list(
    # best_preds = best_preds,
    algo = mff.method,
    eval.method = eval.method,
    memberships = best_membership,
    memberships_rounded = round(best_membership,3),
    best_method = best_method,
    best_params = best_params,
    best_cluster = which.min(best_scores[[eval.method]]),
    best_weight = best_weight,
    best_scores = best_scores
  )

  return(out)
}
