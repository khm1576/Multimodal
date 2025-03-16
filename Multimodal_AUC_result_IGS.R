library(data.table)
library(glmnet)
library(dplyr)
library(pROC)
library(ROCR)
library(xgboost)
library(caret)

# Read data
dat <- fread('/storage0/lab/khm1576/Workspace/disease/Glaucoma_All_Cov.txt')
igs <- fread('/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS.txt')

# Combine datasets and read OCT IDs
dat1 <- cbind(dat[ , -c('app','townsend')], igs[ , -1])
id <- fread('/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt')

# Save original data to re-use for both filtering cases
original_dat1 <- copy(dat1)

# Define filtering cases: 
# "400k": use rows NOT in id$V1 (400k case)
# "55k": use rows that ARE in id$V1 (55k case)
filter_cases <- c("400k", "55k")

# Define modeling scenarios
scenarios <- c("Cov", "PRS", "IGSs", "PRS+IGSs", "PRS+IGSs+Cov")

for (fc in filter_cases) {
  if (fc == "400k") {
    dat1_case <- original_dat1[!(filter %in% id$V1) & !is.na(Gla), ]
  } else if (fc == "55k") {
    dat1_case <- original_dat1[(filter %in% id$V1) & !is.na(Gla), ]
  }
  
  for (scenario in scenarios) {
    if (scenario == "Cov") {
      factor_cols <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")
      dat1_encoded <- model.matrix(~ . - 1, data = dat1_case[, ..factor_cols])
      dat_cat <- cbind(dat1_case[, c(4,5)], dat1_encoded)
      train_mat <- as.matrix(dat_cat)
    } else if (scenario == "PRS") {
      train_mat <- as.matrix(dat1_case[, c(3)])
    } else if (scenario == "IGSs") {
      train_mat <- as.matrix(dat1_case[, -c(1:11)])
    } else if (scenario == "PRS+IGSs") {
      train_mat <- as.matrix(dat1_case[, -c(1,2,4:11)])
    } else if (scenario == "PRS+IGSs+Cov") {
      factor_cols <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")
      dat1_encoded <- model.matrix(~ . - 1, data = dat1_case[, ..factor_cols])
      dat_cat <- cbind(dat1_case[, -c(1,2,6:11)], dat1_encoded)
      train_mat <- as.matrix(dat_cat)
    }
    
    dtrain <- xgb.DMatrix(data = train_mat, label = dat1_case$Gla)
    
    set.seed(123)
    model_cv <- xgb.cv(
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
      prediction = TRUE
    )
    
    predictions <- model_cv$pred
    roc_obj <- roc(dat1_case$Gla, predictions)
    auc_value <- auc(roc_obj)
    cat("Filter case:", fc, "Scenario:", scenario, "AUC:", auc_value, "\n")
  }
}
