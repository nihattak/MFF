#' Train Bagging for Multiple Linear Regression and Produce Model Predictions
#'
#' @description
#' It generates prediction matrices for both validation and test sets by aggregating results
#' from all bootstrap iterations.
#'
#' @details
#' Splits the input dataset into training, validation, and test sets, then performs
#' bootstrap sampling on the training set to fit multiple linear models.
#' Predictions are returned as matrices with dimension \eqn{N_{valid} \times B} and \eqn{N_{test} \times B}.
#' These matrices are the standard input \eqn{x}{x} for \code{mff()} and \code{tune.mff()}.
#'
#' @param target A character string specifying the name of the response variable.
#' @param data A \code{data.frame} containing the target and predictor variables.
#' @param ntest An integer specifying the number of observations to be assigned to the test set.
#' @param nvalid An integer specifying the number of observations to be assigned to the validation set.
#' @param B An integer specifying the number of bootstrap samples to generate.
#' @param seed An optional numeric value for reproducibility.
#' @param parallel Logical; if \code{TRUE}, bootstrap iterations are performed in parallel.
#' Requires \code{doParallel} and \code{foreach} packages.
#' @param ncores An integer specifying the number of CPU cores to use for parallel processing.
#' Defaults to \code{parallel::detectCores() - 1}.
#'
#' @return A list containing the following components:
#' \itemize{
#'   \item \code{pred_matrix_valid}: A matrix of dimensions \eqn{N_{valid} \times B} containing validation set predictions.
#'   \item \code{pred_matrix_test}: A matrix of dimensions \eqn{N_{test} \times B} containing test set predictions.
#'   \item \code{y_valid}: A numeric vector of actual target values for the validation set.
#'   \item \code{y_test}: A numeric vector of actual target values for the test set.
#'   \item \code{metadata}: A list containing the number of training samples (\code{ntrain}),
#'   the number of bootstrap iterations (\code{B}), and parallel processing status.
#' }
#' @examples
#' results <- boot.train(target = "medv", data = MASS::Boston,
#'                            ntest = 50, nvalid = 50, B = 10,
#'                            seed = 123, parallel = FALSE)
#'
#' # Accessing validation prediction matrix
#' head(results$pred_matrix_valid)
#'
#' @import stats
#' @importFrom foreach foreach %dopar% %do%
#' @importFrom parallel makeCluster stopCluster detectCores
#' @importFrom doParallel registerDoParallel
#' @export
boot.train <- function(target, data, ntest, nvalid, B, seed = NULL, parallel = FALSE, ncores = NULL) {

  # --- 1. CRITICAL CHECKS ---
  # A. Missing Argument Check
  if (missing(target) || missing(data) || missing(ntest) || missing(nvalid) || missing(B)) {
    stop("Error: 'target', 'data', 'ntest', 'nvalid', and 'B' must all be provided.")
  }

  # B. Data Type and Content Check
  if (!is.data.frame(data)) stop("Error: 'data' must be a data.frame.")
  if (nrow(data) == 0) stop("Error: The provided dataset is empty.")

  # C. Target Existence and Type Check
  if (!(target %in% names(data))) {
    stop(paste("Error: Target variable '", target, "' not found in the dataset."))
  }
  if (!is.numeric(data[[target]])) {
    warning("Warning: Target variable is not numeric. Ensure this is intended for 'lm'.")
  }

  # D. Split Size Logic
  n <- nrow(data)
  if (ntest < 0 || nvalid < 0) stop("Error: ntest and nvalid must be non-negative integers.")

  total_split <- ntest + nvalid
  if (total_split >= n) {
    stop(sprintf("Error: ntest + nvalid (%d) exceeds or equals total rows (%d). No data left for training.", total_split, n))
  }

  if ((n - total_split) < 5) {
    warning("Warning: Training set will have fewer than 5 observations. Results might be unstable.")
  }

  # E. Bootstrap (B) Logic
  if (B <= 0) {
    stop("Error: The number of bootstrap samples (B) must be a positive integer. You provided B = ", B)
  }

  # F. Missing Values (NA) Check
  if (any(is.na(data))) {
    warning("Warning: Dataset contains NA values. 'lm' will omit these rows by default.")
  }

  # --- 2. DATA SPLITTING ---
  if (!is.null(seed)) set.seed(seed)

  valid_index <- sample(seq_len(n), size = nvalid)
  remaining   <- setdiff(seq_len(n), valid_index)
  actual_ntest <- min(length(remaining), ntest)
  test_index   <- sample(remaining, size = actual_ntest)
  train_index  <- setdiff(remaining, test_index)

  train_data <- data[train_index, ]
  valid_data <- data[valid_index, ]
  test_data  <- data[test_index, ]

  ntrain <- nrow(train_data)
  formula_obj <- as.formula(paste(target, "~ ."))

  # --- 3. HELPER FOR COMBINING ---
  # Bu fonksiyon çekirdeklerden gelen listeleri eleman bazında cbind yapar
  combine_results <- function(...) {
    # Gelen tüm argümanları bir listeye topla
    args <- list(...)

    # Her bir listenin içindeki 'valid' ve 'test' parçalarını ayıkla ve cbind ile birleştir
    list(
      valid = do.call(cbind, lapply(args, `[[`, "valid")),
      test  = do.call(cbind, lapply(args, `[[`, "test"))
    )
  }

  # --- 4. PARALLEL SETUP ---
  `%op%` <- if (parallel) `%dopar%` else `%do%`
  if (parallel) {
    if (is.null(ncores)) ncores <- parallel::detectCores() - 1
    cl <- makeCluster(ncores)
    registerDoParallel(cl)
    on.exit(stopCluster(cl))
  }

  # --- 5. BOOTSTRAP & MODEL TRAINING ---
  # .combine ve .multicombine parametrelerine dikkat!

  i <- NULL

  final_results <- foreach(i = 1:B,
                           .combine = combine_results,
                           .multicombine = TRUE,
                           .packages = "stats") %op% {

                             if (!is.null(seed)) set.seed(seed + i)

                             boot_indices <- sample(seq_len(ntrain), replace = TRUE)
                             boot_train   <- train_data[boot_indices, ]

                             fit_lm <- lm(formula_obj, data = boot_train)

                             # Her iterasyon sadece bu listeyi döner
                             list(
                               valid = predict(fit_lm, valid_data),
                               test  = predict(fit_lm, test_data)
                             )
                           }

  # --- 6. OUTPUT ---
  return(list(
    pred_matrix_valid = final_results$valid,
    pred_matrix_test  = final_results$test,
    y_valid = valid_data[[target]],
    y_test  = test_data[[target]],
    metadata = list(ntrain = ntrain, B = B, parallel = parallel)
  ))
}
