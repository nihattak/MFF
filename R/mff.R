#' Generate Meta-Fuzzy Function
#'
#' @description
#' Construct meta-fuzzy functions by computing membership weights and cluster-wise
#' predictions.
#'
#' @param x A numeric matrix of base-model predictions.
#' @param y A numeric vector of true response values.
#' @param c An integer specifying the number of clusters (functions).
#' @param m A numeric fuzziness exponent (typically m = 2) used in FCM-type membership
#' estimation. Larger values increase fuzziness (more diffuse memberships), while values closer to 1 yield
#' sharper assignments.
#' @param eta numeric regularization parameter used by the possibilistic FCM method (method =
#' "pfcm").
#' @param iter.max An integer specifying the maximum number of iterations allowed for the clustering
#' algorithm.
#' @param nstart n integer controlling the number of random initializations used when method =
#' "kmeans" to improve robustness of the final clustering solution
#' @param stand Logical; if \code{TRUE}, the transposed data is standardized
#' (mean=0, sd=1) before clustering to ensure scale-invariance between models.
#' @param method A character string selecting the membership-generation method. Available options
#' are "fcm", "gk", "pfcm", and "kmeans".
#'
#' @details
#' The \emph{mff} function is the core constructor of the Meta Fuzzy Function (MFF) framework. It
#' takes a matrix of base-model predictions and derives a membership-weight structure
#' that defines multiple meta fuzzy functions using a selected membership-generation method. In
#' the MFF setting, each base learner is represented by its prediction vector across samples;
#' therefore, \emph{mff} internally transposes the prediction matrix so that base models are treated as
#' observations in the meta-clustering space. The resulting membership matrix is then used to
#' form meta fuzzy function predictions through weighted aggregation of base-model outputs.
#'
#' The function supports four membership-generation methods: classical Fuzzy C-Means
#' (FCM), Gustafson--Kessel (GK) clustering for anisotropic structures, possibilistic FCM (PFCM) producing softmax-like weights, and deterministic k-means converted to pseudo-
#' fuzzy memberships. After membership estimation, meta fuzzy function predictions are
#' computed via linear combinations of base-model predictions and the learned membership-
#' based weights, and the predictive performance of each meta fuzzy function is assessed using
#' \emph{evaluate}. Membership weights are standardized column-wise to ensure that the total
#' contribution of base models within each meta fuzzy function sums to one, facilitating
#' interpretation and comparison across meta fuzzy functions.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{method}: The clustering method used for membership estimation.
#'   \item \code{weights}: A column-standardized membership (weight) matrix.
#'   \item \code{cluster_scores}: A data frame of evaluation metrics computed for each cluster(function).
#' }
#'
#' @references
#' Tak, N. (2018). Meta fuzzy functions: Application of recurrent type-1 fuzzy functions.
#' \emph{Applied Soft Computing}, 73, 1-13. \doi{10.1016/j.asoc.2018.08.009}
#'
#' Bezdek, J. C., Ehrlich, R., & Full, W. (1984). FCM: The fuzzy c-means clustering algorithm.
#' \emph{Computers & Geosciences}, 10(2), 191-203. \doi{10.1016/0098-3004(84)90020-7}
#'
#' Cebeci, Z. (2019). Comparison of internal validity indices for fuzzy clustering.
#' \emph{Journal of Agricultural Informatics}, 10(2), 1-14. \doi{10.17700/jai.2019.10.2.537}
#'
#' Gustafson, D. E., & Kessel, W. C. (1978). Fuzzy clustering with a fuzzy covariance matrix.
#' In \emph{1978 IEEE Conference on Decision and Control including the 17th Symposium on Adaptive Processes},
#' 761-766. \doi{10.1109/CDC.1978.268028}
#'
#' Meyer, D., Dimitriadou, E., Hornik, K., Weingessel, A., & Leisch, F. (2024).
#' \emph{e1071: Misc Functions of the Department of Statistics, Probability Theory Group (Formerly: E1071), TU Wien}.
#' R package version 1.7-16. \url{https://CRAN.R-project.org/package=e1071}
#'
#' Gustafson, D. E., & Kessel, W. C. (1978). Fuzzy clustering with a fuzzy covariance matrix.
#' In \emph{1978 IEEE Conference on Decision and Control including the 17th Symposium on Adaptive Processes},
#' 761-766. \doi{10.1109/CDC.1978.268028}
#'
#' Pal, N. R., Pal, K., Keller, J. M., & Bezdek, J. C. (2005). A possibilistic fuzzy c-means clustering algorithm.
#' \emph{IEEE Transactions on Fuzzy Systems}, 13(4), 517-530. \doi{10.1109/TFUZZ.2004.840099}
#'
#'
#' @seealso
#' \code{\link{model.train}} for preparing input matrices,
#' \code{\link{predict.mff}} for test set predictions,
#' \code{\link{tune.mff}} for hyperparameter optimization,
#' \code{\link{evaluate}} for performance metrics.
#'
#'
#' @examples
#'  result_train <- model.train(
#'     target = "medv",
#'     data = MASS::Boston,
#'     ntest = 50,
#'     nvalid = 50,
#'     seed = 123
#'  )
#'
#'  mff_model <- mff(result_train$pred_matrix_valid, result_train$y_valid, c = 4,
#'  iter.max=100,nstart = 100,method = "kmeans")
#'  mff_model
#'
#' @export
mff <- function(x, y, c, m=2, eta=2,iter.max=NULL,nstart = 1, stand = FALSE,method = c("fcm", "pfcm", "kmeans","gk")) {

  if (any(is.na(x))) stop("Input matrix 'x' contains NA values. Please impute or remove them.")

  col_vars <- apply(x, 2, var)
  if (any(col_vars == 0)) {
    warning("Constant columns detected in 'x'. These may cause singularity.")
  }

  if (c > ncol(x)) {
    stop(sprintf(
      "Number of clusters (%d) cannot exceed the number of models (%d).",
      c, ncol(x)
    ))
  }

  method <- match.arg(method)

  if (is.null(iter.max)) {
    iter.max <- switch(method,
                       "kmeans" = 10,
                       "fcm"    = 100,
                       "pfcm"   = 1000,
                       "gk"     = 1000)
  }

  x_transposed <- t(x)
  if (stand) {
    x_processed <- scale(x_transposed)
  } else {
    x_processed <- x_transposed
  }
  if (method == "fcm") {
    result <- e1071::cmeans(x_processed, centers = c, m = m,iter.max=iter.max)
    membership <- result$membership
  } else if (method == "pfcm") {
    result <- ppclust::pfcm(x_transposed, centers = c, m = m, nstart = nstart, eta = eta,stand = stand,iter.max=iter.max)
    membership <- result$u
  }else if (method == "gk") {

    first_warning_caught <- FALSE

    result <- tryCatch({
      withCallingHandlers({
        ppclust::gk(x_transposed, centers = c, m = m, nstart = nstart, stand = stand, iter.max = iter.max)
      }, warning = function(w) {
        if (!first_warning_caught) {

          call_info <- deparse(w$call)

          message(paste0("Note: GK caught a numerical warning in [", call_info, "] -> ", w$message))

          first_warning_caught <<- TRUE
        }
        invokeRestart("muffleWarning")
      })
    }, error = function(e) {
      stop(paste("GK algorithm failed due to numerical instability:", e$message))
    })

    membership <- result$u

    if(any(is.na(membership))) {
      membership[is.na(membership)] <- 1/c

      warning("Numerical instability detected in GK: Some NaNs were replaced with uniform weights.")
    }

    }else if (method == "kmeans") {
    result <- stats::kmeans(x_processed, centers = c, nstart = nstart, iter.max = iter.max)
    membership <- .weight_kmeans(result$cluster)
  } else {
    stop("Unknown method.")
  }

  weight_matrix <- prop.table(membership, margin = 2)
  rownames(weight_matrix) <- colnames(x)

  cluster_preds <- x %*% weight_matrix
  cluster_scores <- evaluate(cluster_preds, y)

  out <- list(
    method = method,
    weights = weight_matrix,
    cluster_scores = cluster_scores
  )

  out <- structure(out, class = "mff")

  return(out)
}
