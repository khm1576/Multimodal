library(data.table)
library(xgboost)
library(pROC)

dat <- fread('/storage0/lab/khm1576/Workspace/disease/Glaucoma_All_Cov.txt')
ids <- fread('/storage0/lab/khm1576/Image/OCT/IDS.txt')
oct <- fread('/storage0/lab/khm1576/IDPs/OCT/OCT_IDPs.txt')
setnames(oct, old = names(oct), new = gsub("-0\\.0$", "", names(oct)))
id <- fread('/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt')

dat1 <- cbind(dat[(filter %in% id$V1), -c('app','townsend')], ids[,-1])
dat1 <- cbind(dat1, oct[(filter %in% id$V1), -1])

factor_cols <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")
dat1_encoded <- model.matrix(~ . - 1, data = dat1[, ..factor_cols])

scenarios <- c("Cov", "PRS", "IDPs", "IDS", "PRS+IDS", "PRS+IDS+Cov+IDPs")

for (scenario in scenarios) {
  if (scenario == "Cov") {
    dat_cat <- cbind(dat1[, c(4, 5)], dat1_encoded)
    train_mat <- as.matrix(dat_cat)
  } else if (scenario == "PRS") {
    train_mat <- as.matrix(dat1[, c(3)])
  } else if (scenario == "IDS") {
    train_mat <- as.matrix(dat1[, c(12)])
  } else if (scenario == "IDPs") {
    train_mat <- as.matrix(dat1[, -c(1:12)])
  } else if (scenario == "PRS+IDS") {
    train_mat <- as.matrix(dat1[, c(3, 12)])
  } else if (scenario == "PRS+IDS+Cov+IDPs") {
    dat_cat <- cbind(dat1[, -c(1,2,6:11,12)], dat1_encoded)
    train_mat <- as.matrix(dat_cat)
  }
  
  dtrain <- xgb.DMatrix(data = train_mat, label = dat1$Gla)
  
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
  roc_obj <- roc(dat1$Gla, predictions)
  auc_value <- auc(roc_obj)
  cat(scenario, "AUC:", auc_value, "\n")
}


