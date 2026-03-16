library(data.table)
library(glmnet)
library(dplyr)
library(pROC)
library(ROCR)
library(xgboost)
library(caret)
library(ModelMetrics)
library(SHAPforxgboost)



#Loading Data
dat <- fread('/storage0/lab/khm1576/연구주제/disease/Glaucoma_All_Cov.txt')
igs <- fread('/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS.txt')
igs2 <- fread('/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS2.txt')
setkey(igs, app14048)
setkey(igs2, app14048)
igs <- merge(igs, igs2, by = "app14048", all = TRUE)
dat1 <- cbind(dat[,-c('app77890','townsend')],igs[,-1])
id <- fread('/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt')
dat1 <- dat1[!(app14048 %in% id$V1)&!is.na(dat1$Gla),]   #440k
#dat1 <- dat1[(app14048 %in% id$V1)&!is.na(dat1$Gla),]   #55k




#mutimodal xgboost
factor_cols <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")
#dat1_encoded <- model.matrix(~ . - 1, data = dat1[, ..factor_cols])                 
dat_cat <- cbind(dat1_clean[, c(4,5)], dat1_encoded)
dat1_clean <- dat1[complete.cases(dat1[, ..factor_cols]), ]
dat1_encoded <- model.matrix(~ . - 1, data = dat1_clean[, ..factor_cols])
dat_cat <- cbind(dat1_clean[, c(3,4,5,12:57)], dat1_encoded)
colSums(is.na(dat_cat))
train_mat <- as.matrix(dat_cat)
dtrain <- xgb.DMatrix(data = train_mat, label = dat1_clean$Gla)
dtrain
dat1_clean
colnames(train_mat)
colnames(dtrain)

set.seed(123)
# 모델 학습
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
roc_obj <- roc(dat1_clean$Gla, predictions)
ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
print(ci)


lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
auc_value <- pROC::auc(roc_obj)
cat("AUC : ",auc_value)
cat("AUC DeLong SE:", round(se_est, 5), "\n")
y <- dat1_clean$Gla
p <- predictions
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")

#SHAP
best_iter <- model$best_iteration
best_iter
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


my_dt <- shap_result$mean_shap_score

my_dt2<-as.data.frame(my_dt)
colnames(my_dt2) <-  c("mean_shap_score")
my_dt2


prs_vars <- c("PRS_scale")  
igs_vars <- rownames(my_dt2)[grepl("^\\d+$", rownames(my_dt2))]  
clinical_vars <- setdiff(rownames(my_dt2), c(prs_vars, igs_vars))  


prs_sum <- sum(my_dt2[prs_vars, "mean_shap_score"], na.rm = TRUE)
igs_sum <- sum(my_dt2[igs_vars, "mean_shap_score"], na.rm = TRUE)
clinical_sum <- sum(my_dt2[clinical_vars, "mean_shap_score"], na.rm = TRUE)


df_shap_grouped <- data.frame(
  Group = c("PRS", "IGS", "Clinical"),
  SHAP_Importance = c(prs_sum, igs_sum, clinical_sum)
)


shap_result$mean_shap_score
print(df_shap_grouped)















k2 <- as.data.frame(shap_result$mean_shap_score)
k2




k2_vec <- setNames(k2[[1]], rownames(k2))


fixed_vars <- c("PRS_scale", "age", "sex")
numeric_vars <- names(k2_vec)[grepl("^\\d", names(k2_vec))]


illness_vars  <- grep("^illness", names(k2_vec), value = TRUE)
centre_vars   <- grep("^centre", names(k2_vec), value = TRUE)
smoke_vars    <- grep("^smoke", names(k2_vec), value = TRUE)
edu_vars      <- grep("^edu", names(k2_vec), value = TRUE)
alcohol_vars  <- grep("^alcohol", names(k2_vec), value = TRUE)
ethnic_vars   <- grep("^ethnic|^27855$|^27857$|^centre11002$|^centre11003$", names(k2_vec), value = TRUE)


illness_sum  <- sum(k2_vec[illness_vars])
centre_sum   <- sum(k2_vec[centre_vars])
smoke_sum    <- sum(k2_vec[smoke_vars])
edu_sum      <- sum(k2_vec[edu_vars])
alcohol_sum  <- sum(k2_vec[alcohol_vars])
ethnic_sum   <- sum(k2_vec[ethnic_vars])


fixed_values <- k2_vec[fixed_vars]
numeric_values <- k2_vec[numeric_vars]


final_result <- c(
  fixed_values["PRS_scale"],
  numeric_values,
  fixed_values[c("age", "sex")],
  illness = illness_sum,
  centre = centre_sum,
  smoke = smoke_sum,
  edu = edu_sum,
  alcohol = alcohol_sum,
  ethnic = ethnic_sum
)
print(final_result)


class(final_result)

k_df <- as.data.frame(final_result)



