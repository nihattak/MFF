#' Hyperparameter Search for Meta-Fuzzy Function
#'
#' @description
#' The \emph{tune.mff} function performs hyperparameter optimization via grid search for Meta Fuzzy Functions
#' (MFFs) by searching over clustering-related parameter combinations and selecting the configuration that
#' yields the lowest validation error.
#'
#' @param x A numeric matrix of base-model predictions with dimensions \eqn{N_{test} \times M}. Each
#' column corresponds to a base learner.
#' @param y numeric vector of validation targets. This vector is used to evaluate meta fuzzy function
#' predictions.
#' @param max_c An integer specifying the maximum number of clusters to be considered in the
#' search.
#' @param m_seq A numeric vector of candidate values for the fuzziness exponent m used in FCM-
#' type methods.
#' @param eta_seq A numeric vector of candidate values for the probabilistic regularization parameter
#' \eqn{\eta}{eta} used when mff.method = "pfcm".
#' @param iter.max An integer specifying the maximum number of iterations allowed for the clustering
#' algorithm within each grid evaluation..
#' @param nstart integer; An integer controlling the number of random initializations for k-means
#' when mff.method = "kmeans".
#' @param seed An integer used to set the random seed for reproducibility during weight computation
#' and parameter search.
#' @param mff.method A character string selecting the membership-generation method.
#' @param eval.method A character string specifying the metric used to select the best-performing
#' meta fuzzy function.
#' @param stand Logical; if \code{TRUE}, the transposed data is standardized
#' (mean=0, sd=1) before clustering to ensure scale-invariance between models.
#' @param parallel Logical; if \code{TRUE}, the grid search process is executed in parallel
#' to accelerate hyperparameter optimization.
#' @param num_cores An integer specifying the number of CPU cores to utilize for
#' parallel processing. If \code{NULL} (default), the function automatically detects
#' and uses all available cores minus one (\code{parallel::detectCores() - 1}).
#' @param logging A logical flag indicating whether progress information is printed during the search.
#'
#' @details
#' Given a matrix of base-model predictions and the corresponding validation targets, \emph{tune.mff}
#' repeatedly calls \emph{mff} to compute membership weights, generate meta fuzzy function
#' predictions, and evaluate these predictions using a user-specified metric. The best
#' configuration is determined by the minimum value of the selected evaluation metric among
#' the scores obtained from the meta fuzzy function predictions produced under each candidate
#' setting.
#'
#' The search space depends on the selected membership-generation method. For classical Fuzzy
#' C-Means ("fcm"), the function explores combinations of the number of clusters c and the fuzziness index m. For possibilistic FCM ("pfcm"), the
#' grid additionally includes the possibilistic regularization parameter \eqn{\eta}{eta}. For k-means
#' ("kmeans"), the search is performed only over the number of clusters(c). The function returns
#' the best-performing configuration together with the corresponding weight structure, the index
#' of the best-performing meta fuzzy function, and the full set of evaluation results, enabling
#' transparent reporting and reproducible model selection.
#'
#' @return
#' \itemize{
#'   \item \code{algorithm}: The selected membership-generation method.
#'   \item \code{eval.method}: The evaluation metric used in model selection.
#'   \item \code{weights}: The membership (weight) matrix associated with the best-performing
#'   configuration.
#'   \item \code{best_params}: A list containing the hyperparameters that achieved the best score.
#'   \item \code{best_cluster}: The index of the meta fuzzy function yielding the minimum validation error.
#'   \item \code{best_weight}: The weight vector corresponding to the best-performing meta fuzzy function.
#'   \item \code{best_scores}: The full set of evaluation scores for all meta fuzzy function predictions under
#'   the best configuration.
#' }
#'
#' @seealso \code{\link{mff}}, \code{\link{model.train}}, \code{\link{predict.mff}}
#'
#' @examples
#'   res <- model.train(target="medv", data=MASS::Boston, ntest=50, nvalid=50, seed = 123)
#'   fit <- tune.mff(res$pred_matrix_valid, res$y_valid, max_c=6, mff.method="kmeans")
#'   out <- predict(fit, pred_matrix=res$pred_matrix_test, type="best")
#'   head(out$mff_preds)
#'   out$mff_weights
#'
#' @importFrom foreach foreach %dopar%
#' @importFrom parallel makeCluster stopCluster detectCores
#' @importFrom doParallel registerDoParallel
#' @export
tune.mff <- function(x, y, max_c, m_seq = seq(1.1, 3, by = 0.1),
                     eta_seq = seq(1.1, 3, by = 0.4), iter.max = NULL,
                     nstart = 1, seed = 123,
                     mff.method = c("fcm", "pfcm", "kmeans", "gk"),
                     eval.method = c("MAE", "RMSE", "MAPE", "SMAPE", "MSE", "MedAE"),
                     stand = FALSE, parallel = FALSE, num_cores = NULL, logging = TRUE) {

  if (max_c > ncol(x)) {
    stop(sprintf("Number of clusters (%d) cannot exceed models (%d).", max_c, ncol(x)))
  }

  mff.method <- match.arg(mff.method)
  eval.method <- match.arg(eval.method)

  # --- 1. Grid Construction ---
  if (mff.method == "pfcm") {
    search_grid <- expand.grid(m = m_seq, c = 2:max_c, eta = eta_seq,
                                nstart = nstart, KEEP.OUT.ATTRS = FALSE)
  } else if (mff.method == "fcm") {
    search_grid <- expand.grid(m = m_seq, c = 2:max_c, KEEP.OUT.ATTRS = FALSE)
  } else if (mff.method == "gk") {
    search_grid <- expand.grid(m = m_seq, c = 2:max_c,
                               nstart = nstart, KEEP.OUT.ATTRS = FALSE)
  } else if (mff.method == "kmeans") {
    search_grid <- expand.grid(c = 2:max_c, nstart = nstart, KEEP.OUT.ATTRS = FALSE)
  }

  num_combinations <- nrow(search_grid)
  if(logging) cat("Number of Combinations:", num_combinations, "\n")

  # --- 2. Parallel Setup ---
  if (parallel) {
    if (!requireNamespace("doParallel", quietly = TRUE)) stop("Package 'doParallel' is required for parallel processing.")
    if (!requireNamespace("foreach", quietly = TRUE)) stop("Package 'foreach' is required for parallel processing.")

    if (is.null(num_cores)) num_cores <- parallel::detectCores() - 1
    cl <- parallel::makeCluster(num_cores)
    doParallel::registerDoParallel(cl)
    on.exit(parallel::stopCluster(cl)) # Ensure cluster stops even if function fails

    if(logging) cat("Running in parallel across", num_cores, "cores...\n")
  }

  # --- 3. Execution ---
  # We use a helper to wrap the logic so it works for both serial and parallel
  run_iteration <- function(i) {
    set.seed(seed + i) # Ensure each worker has a unique but reproducible seed
    params <- as.list(search_grid[i, , drop = FALSE])

    mff_result <- mff(
      x = x, y = y, c = params$c,
      m = params$m, eta = params$eta,
      iter.max = iter.max, nstart = params$nstart,
      method = mff.method,
      stand = stand
    )

    # Return metric and result
    metric <- min(mff_result$cluster_scores[, eval.method])
    return(list(metric = metric, result = mff_result, params = params))
  }

  if (parallel) {
    # Parallel loop using foreach
    all_results <- foreach::foreach(i = 1:num_combinations,
                                    .packages = c("e1071", "ppclust", "stats"),
                                    .export = c("mff", "evaluate", ".weight_kmeans")) %dopar% {
      run_iteration(i)
    }
  } else {
    # Standard serial loop
    all_results <- list()
    for (i in 1:num_combinations) {
      if(logging) cat(i, " ")
      all_results[[i]] <- run_iteration(i)
    }
    if(logging) cat("\n")
  }

  # --- 4. Best Model Selection ---
  metrics <- sapply(all_results, function(res) res$metric)
  best_idx <- which.min(metrics)
  best_res <- all_results[[best_idx]]$result

  idx_in_cluster <- unname(which.min(best_res$cluster_scores[, eval.method]))
  best_weight <- best_res$weights[, idx_in_cluster]

  out <- list(
    algorithm = mff.method,
    eval.method = eval.method,
    weights = best_res$weights,
    best_params = all_results[[best_idx]]$params,
    best_cluster = idx_in_cluster,
    best_weight = best_weight,
    best_scores = best_res$cluster_scores
  )

  out <- structure(out, class = "mff")
  return(out)
}
