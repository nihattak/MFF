#' MFF: Meta-Fuzzy Functions
#'
#' The MFF package provides an integrated framework for fuzzy clustering-based
#' regression, ensemble modeling, and performance evaluation. It includes
#' methods such as PFCM, GK-based clustering, K-Means and C-Means clustering.
#'
#' Core features include:
#' \itemize{
#'   \item Fit a fuzzy-cluster–based meta-ensemble model.
#'   \item Ensemble regression models (LM, Ridge, Lasso, Elastic Net, RF, XGBoost, LightGBM)
#'   \item Model comparison and evaluation utilities (MAE, RMSE, MAPE, SMAPE, MSE, MedAE)
#' }
#'
#' @section Available Functions:
#' \describe{
#'   \item{\code{mff()}}{Fit a fuzzy-cluster–based meta-ensemble model.}
#'   \item{\code{tune.mff()}}{Hyperparameter tuning for meta-fuzzy models.}
#'   \item{\code{predict.mff()}}{Generate predictions using cluster-based weights. S3 method for predict()}
#'   \item{\code{evaluate()}}{Compute evaluation metrics (MAE, RMSE, MAPE, SMAPE, MSE, MedAE).}
#'   \item{\code{model.train()}}{Train regression models (LM, Ridge, Lasso, Elastic Net, RF, XGBoost, LightGBM).}
#' }
#'
#' @section Methods:
#' The package implements meta-fuzzy functions that derive weights from fuzzy
#' membership matrices and use them to construct a weighted ensemble of
#' multiple base models.
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end
NULL
