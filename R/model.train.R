#' Train Multiple Regression and ML Models and Produce Model Predictions
#'
#' This function trains several machine learning models (LM, Lasso, Ridge,
#' Elastic Net, Random Forest, XGBoost, LightGBM,and returns prediction matrices for validation and
#' test sets. It is designed for stacking/ensemble meta-learning workflows.
#'
#' The function randomly splits the dataframe into training, validation, and
#' test sets based on given sample sizes. Each model is trained on the training
#' set, and predictions are generated for validation and test sets. These
#' predictions form the model prediction meta-features for stacking models.
#'
#' @param target Character string. Name of the target column to predict.
#' @param df A data frame containing the predictors and the target variable.
#' @param test_count Integer. Number of observations to allocate to the test set.
#' @param valid_count Integer. Number of observations to allocate to the validation set.
#'
#' @details
#' **Models trained**
#' \itemize{
#'   \item Linear Regression (LM)
#'   \item Lasso Regression (glmnet, alpha = 1)
#'   \item Ridge Regression (glmnet, alpha = 0)
#'   \item Elastic Net (glmnet, alpha = 0.5)
#'   \item Random Forest (100 trees)
#'   \item XGBoost (reg:squarederror loss)
#'   \item LightGBM (regression objective)
#' }
#'
#' Validation and test predictions are combined into matrices used for stacking:
#' \itemize{
#'   \item \code{pred_matrix_valid}: Validation predictions
#'   \item \code{pred_matrix_test}: Test predictions
#' }
#'
#'
#' @return A list containing:
#' \describe{
#'   \item{pred_matrix_valid}{Matrix of model predictions for validation set.}
#'   \item{pred_matrix_test}{Matrix of model predictions for test set.}
#'   \item{y_test}{True test labels as numeric vector.}
#'   \item{y_valid}{True validation labels as numeric vector.}
#' }
#'
#' @examples
#' \dontrun{
#'   result <- model.train(
#'     target = "SalePrice",
#'     df = housing_df,
#'     test_count = 300,
#'     valid_count = 300
#'   )
#'
#'   head(result$pred_matrix_valid)
#' }
#'
#' @import glmnet
#' @import randomForest
#' @import xgboost
#' @import lightgbm
#'
#' @export
model.train <- function(target,df,test_count,valid_count) {
n <- nrow(df)

set.seed(123)

# Validation set
valid_index <- sample(seq_len(n), size = valid_count)
remaining <- setdiff(seq_len(n), valid_index)

# Test set
test_index <- sample(remaining, size = test_count)

# Train set
train_index <- setdiff(remaining, test_index)

# Create sets
train_data <- df[train_index, ]
valid_data <- df[valid_index, ]
test_data  <- df[test_index, ]

# Training models
# Formula for models (no feature selection)
formula <- as.formula(paste(target, "~ ."))

# Linear model
lm <- lm(formula, data = train_data)
lm_pred_valid <- predict(lm,valid_data)
lm_pred_test <- predict(lm,test_data)

# Lasso Ridge Elastic Net
X_train <- model.matrix(formula, train_data)[, -1]
y_train <- as.matrix(train_data[target])

X_valid <- model.matrix(formula, valid_data)[, -1]
y_valid <- as.matrix(valid_data[target])

X_test <- model.matrix(formula, test_data)[, -1]
y_test <- as.matrix(test_data[target])

# Lasso
cv_model_lasso <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)
lasso_pred_valid <- predict(cv_model_lasso,s = "lambda.min" ,X_valid)
lasso_pred_test <- predict(cv_model_lasso,s = "lambda.min" ,X_test)

# Ridge
cv_model_ridge <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)
ridge_pred_valid <- predict(cv_model_ridge,s = "lambda.min" ,X_valid)
ridge_pred_test <- predict(cv_model_ridge,s = "lambda.min" ,X_test)

# Elastic Net
cv_model_elastic <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)
elastic_pred_valid <- predict(cv_model_elastic,s = "lambda.min" ,X_valid)
elastic_pred_test <- predict(cv_model_elastic,s = "lambda.min" ,X_test)

# Random Forest (RF)
rf_model <- randomForest(formula, data = train_data, ntree = 100)
rf_pred_valid <- predict(rf_model,valid_data)
rf_pred_test <- predict(rf_model,test_data)

# XGBoost
nrounds = 200
eta = 0.1
max_depth = 6

xgboost_dtrain <- xgb.DMatrix(X_train, label = y_train)
xgboost_params <- list(objective = "reg:squarederror", eta = eta, max_depth = max_depth, eval_metric = "rmse")
xgboost_model <- xgb.train(xgboost_params, xgboost_dtrain, nrounds = nrounds, verbose = 0)
xgboost_pred_valid <- predict(xgboost_model, xgb.DMatrix(X_valid))
xgboost_pred_test <- predict(xgboost_model, xgb.DMatrix(X_test))

# LightGBM
learning_rate = 0.05
num_leaves = 31

lightgbm_dtrain <- lgb.Dataset(X_train, label = y_train)
lightgbm_params <- list(objective = "regression", metric = "rmse", learning_rate = learning_rate, num_leaves = num_leaves, verbose = -1)
lightgbm_model <- lgb.train(lightgbm_params, lightgbm_dtrain, nrounds = nrounds, verbose = -1)
lightgbm_pred_valid <- predict(lightgbm_model, X_valid)
lightgbm_pred_test <- predict(lightgbm_model, X_test)

# Pred Matrix

pred_matrix_valid <- cbind(lm_pred_valid, lasso_pred_valid, ridge_pred_valid, elastic_pred_valid ,rf_pred_valid,xgboost_pred_valid,lightgbm_pred_valid)
pred_matrix_test <- cbind(lm_pred_test, lasso_pred_test, ridge_pred_test, elastic_pred_test ,rf_pred_test,xgboost_pred_test,lightgbm_pred_test)

modelNames <- c("LM","Lasso","Ridge","ElasticNet","RF","XGBoost","LightGBM")
colnames(pred_matrix_valid) <- modelNames
colnames(pred_matrix_test) <- modelNames

# Function Output

out <- list()
out$pred_matrix_valid <- pred_matrix_valid
out$pred_matrix_test <- pred_matrix_test
out$y_test <- as.numeric(y_test)
out$y_valid <-as.numeric(y_valid)
return(out)
}
