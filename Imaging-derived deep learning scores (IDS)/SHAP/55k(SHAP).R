library(data.table)
library(pROC)
library(dplyr)
library(xgboost)
library(caret)
library(ModelMetrics)
library(SHAPforxgboost)




# Loading Data
file_name ="ConvNext_femto" #file_name is one of "ConvNext_femto", "PIT", "deit", "xcit", "VIT"
ids_path <- paste0("/home/guestuser1/", file_name, "/IDS/IDS(xgboost).txt")
dat <- fread('/storage0/lab/khm1576/연구주제/disease/Glaucoma_All_Cov.txt')
prs <- fread('/storage0/lab/khm1576/연구주제/PRS/Glaucoma_app14048.txt')
ids <- fread(ids_path)
oct <- fread('/storage0/lab/khm1576/IDPs/OCT/OCT_IDPs.txt')
id <- fread('/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt')
oct <- oct[, !c("27851-0.0", "27853-0.0", "27855-0.0", "27857-0.0"), with = FALSE]
setnames(oct, old = names(oct), new = gsub("-0\\.0$", "", names(oct)))
setnames(ids, old = "ID", new = "app77890")
id_map <- unique(dat[, .(app77890, app14048)])
ids <- merge(ids, id_map, by = "app77890", all.x = TRUE)

dat1 <- dat[(app14048 %in% id$V1), -c('townsend'), with = FALSE]
dat1
ids_subset <- ids[, .(app14048, IDS)]
dat1 <- merge(dat1, ids_subset, by = "app14048")
oct_sub <- oct[(app14048 %in% id$V1)]
dat1 <- cbind(dat1, oct_sub[, -1, with = FALSE]) 
dat1 <- dat1[, -2]
factor_cols <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")
dat1_encoded <- model.matrix(~ . - 1, data = dat1[, ..factor_cols])




#multimodal xgboost
dat_cat <- cbind(dat1[,-c(1,2,6:11)],dat1_encoded)
train_mat <- as.matrix(dat_cat)
colnames(train_mat)
dtrain <- xgb.DMatrix(data=train_mat,label=dat1$Gla)

set.seed(123)
# train model
model <- xgb.cv(
  params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 3,
    eta = 0.01,
    alpha = 0.1,
    lambda = 0.5
  ),
  data = dtrain,
  nfold = 5,
  nrounds = 2000,
  early_stopping_rounds = 10,
  prediction = T
)

predictions <- model$pred
roc_obj <- roc(dat1$Gla, predictions)
auc_value <- pROC::auc(roc_obj)
cat("AUC : ",auc_value,"\n\n")

#SHAP
best_iter <- model$best_iteration
final_model <- xgboost(
  params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 3,
    eta = 0.01,
    alpha = 0.1,
    lambda = 0.5
  ),
  data = dtrain,
  nrounds = best_iter,
  verbose = TRUE
)

shap_result <- shap.values(xgb_model = final_model, X_train = train_mat)

shap_result
shap_result$shap_score
sum(shap_result$mean_shap_score)

# shap.prep() returns the long-format SHAP data from either model or
shap_values_iris <- shap_result$shap_score
shap_long_iris <- shap.prep(xgb_model = final_model, X_train = train_mat)
# is the same as: using given shap_contrib
shap_long_iris <- shap.prep(shap_contrib = shap_values_iris, X_train = train_mat)
my_dt <- shap_result$mean_shap_score
my_dt2<-as.data.frame(my_dt)
colnames(my_dt2) <-  c("mean_shap_score")
colnames(my_dt2)
cat("Mean Abosolutely SHAP score")
my_dt2



prs_var <- "PRS_scale"
ids_var <- "IDS"
clinical_vars <- c("age", "illness", "centre", "edu", "alcohol", "smoke", "sex", "ethnic")
idp_vars <- setdiff(rownames(my_dt2), c(prs_var, ids_var, clinical_vars))
prs_var <- "PRS_scale"
ids_var <- "IDS"
clinical_vars <- c("age", "illness", "centre", "edu", "alcohol", "smoke", "sex", "ethnic")
idp_vars <- setdiff(rownames(my_dt2), c(prs_var, ids_var, clinical_vars))
prs_sum <- sum(my_dt2[prs_var, "mean_shap_score"], na.rm = TRUE)
ids_sum <- sum(my_dt2[ids_var, "mean_shap_score"], na.rm = TRUE)
clinical_sum <- sum(my_dt2[clinical_vars, "mean_shap_score"], na.rm = TRUE)
idp_sum <- sum(my_dt2[idp_vars, "mean_shap_score"], na.rm = TRUE)

df_grouped_shap <- data.frame(
  Group = c("PRS", "IDS", "Clinical", "IDPs"),
  SHAP_Sum = c(prs_sum, ids_sum, clinical_sum, idp_sum)
)

print(df_grouped_shap)



