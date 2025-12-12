# devtools::document()
# devtools::build_manual()

library(MFF)
boston <- MASS::Boston
# boston$chas <- factor(boston$chas,levels = c(0,1), labels = c("No","Yes"))
result_train <- model.train(
  target = "medv",
  df = boston,
  test_count = 50,
  valid_count = 50
)

mff_fcm_tuned <- tune.mff(result_train$pred_matrix_valid, result_train$y_valid,
                        max_c = 5, iter.max=1000,nstart = 100,mff.method = "fcm",eval.method = "MAPE")
mff_fcm_tuned

pred_weight <- predict(mff_fcm_tuned,result_train$pred_matrix_test,type = "all")
pred_weight
evaluate(pred_weight$mff_preds,result_train$y_test)

set.seed(123)
mff_fcm_best <- mff(result_train$pred_matrix_valid, result_train$y_valid,
            c = 3,m = 3,method = "fcm")

mff_fcm_best

pred_weight <- predict(mff_fcm_best,result_train$pred_matrix_test)
evaluate(pred_weight$mff_preds,result_train$y_test)

evaluate(result_train$pred_matrix_test,result_train$y_test)

#################################################################################################
#################################################################################################

library(MFF)
library(dplyr)
library(readr)
house <- read_csv("C:/Users/sadik/Downloads/kc_house_data.csv/kc_house_data.csv", locale = locale(encoding = "UTF-8")) %>% na.omit() %>% dplyr::select(-id,-date)
result_train <- model.train(
  target = "price",
  df = house,
  test_count = 2160,
  valid_count = 2160
)

mff_fcm_tuned <- tune.mff(result_train$pred_matrix_valid, result_train$y_valid,
                          max_c = 5, iter.max=100,nstart = 100,mff.method = "fcm",eval.method = "MAPE")
mff_fcm_tuned

pred_weight <- predict(mff_fcm_tuned,result_train$pred_matrix_test)
pred_weight
evaluate(pred_weight$mff_preds,result_train$y_test)

set.seed(123)
mff_fcm_best <- mff(result_train$pred_matrix_valid, result_train$y_valid,
                    c = 4,m = 1.5,method = "fcm")

mff_fcm_best

pred_weight <- predict(mff_fcm_best,result_train$pred_matrix_test,cluster = 1)
evaluate(pred_weight$mff_preds,result_train$y_test)

evaluate(result_train$pred_matrix_test,result_train$y_test)

#################################################################################################
#################################################################################################

result_train <- model.train(
  target = "mpg",
  df = mtcars,
  test_count = 5,
  valid_count = 5
)

mff_fcm_tuned <- tune.mff(result_train$pred_matrix_valid, result_train$y_valid,
                          max_c = 5, iter.max=100,nstart = 100,mff.method = "fcm",eval.method = "MAPE")
mff_fcm_tuned

pred_weight <- predict(mff_fcm_tuned,result_train$pred_matrix_test,"all")
pred_weight
evaluate(pred_weight$mff_preds,result_train$y_test)

set.seed(123)
mff_fcm_best <- mff(result_train$pred_matrix_valid, result_train$y_valid,
                    c = 5,m = 1.1,method = "fcm")

mff_fcm_best

pred_weight <- predict(mff_fcm_best,result_train$pred_matrix_test)
evaluate(pred_weight$mff_preds,result_train$y_test)

evaluate(result_train$pred_matrix_test,result_train$y_test)

#################################################################################################
#################################################################################################

library(AmesHousing)

ames <- make_ames()
Ames <- ames %>% select(where(is.numeric))

result_train <- model.train(
  target = "Sale_Price",
  df = Ames,
  test_count = 293,
  valid_count = 293
)

mff_fcm_tuned <- tune.mff(result_train$pred_matrix_valid, result_train$y_valid,
                          max_c = 5, iter.max=100,nstart = 100,mff.method = "fcm",eval.method = "MAPE")
mff_fcm_tuned

pred_weight <- predict(mff_fcm_tuned,result_train$pred_matrix_test,"all")
pred_weight
evaluate(pred_weight$mff_preds,result_train$y_test)

set.seed(123)
mff_fcm_best <- mff(result_train$pred_matrix_valid, result_train$y_valid,
                    c = 5,m = 3,method = "fcm")

mff_fcm_best

pred_weight <- predict(mff_fcm_best,result_train$pred_matrix_test)
evaluate(pred_weight$mff_preds,result_train$y_test)

evaluate(result_train$pred_matrix_test,result_train$y_test)

#################################################################################################
#################################################################################################

wine_quality <- read.csv("C:/Users/sadik/Downloads/winequality-red.csv",sep =";")

result_train <- model.train(
  target = "quality",
  df = wine_quality,
  test_count = 160,
  valid_count = 160
)

mff_fcm_tuned <- tune.mff(result_train$pred_matrix_valid, result_train$y_valid,
                          max_c = 7, iter.max=1000,nstart = 100,mff.method = "fcm",eval.method = "MAPE")
mff_fcm_tuned

pred_weight <- predict(mff_fcm_tuned,result_train$pred_matrix_test,"best")
pred_weight
evaluate(pred_weight$mff_preds,result_train$y_test)

set.seed(123)
mff_fcm_best <- mff(result_train$pred_matrix_valid, result_train$y_valid,
                    c = 4,m = 2.4,method = "fcm")

mff_fcm_best

pred_weight <- predict(mff_fcm_best,result_train$pred_matrix_test)
evaluate(pred_weight$mff_preds,result_train$y_test)

evaluate(result_train$pred_matrix_test,result_train$y_test)
