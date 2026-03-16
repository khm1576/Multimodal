library(data.table)
library(pROC)
library(dplyr)
library(xgboost)
library(caret)
library(ModelMetrics)
library(iml)


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
colnames(dat1)

################################################################################
factor_cols <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")
dat1_encoded <- model.matrix(~ . - 1, data = dat1[, ..factor_cols])

scenarios <- c("Cov", "PRS", "IDPs", "IDS","PRS+IDPs" ,"PRS+IDS", "PRS+IDS+Cov+IDPs")
colnames(dat1)

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
  } else if (scenario == "PRS+IDPs"){
    train_mat <- as.matrix(dat1[,c(3,13:54)])
  } else if (scenario == "PRS+IDS") {
    train_mat <- as.matrix(dat1[, c(3, 12)])
  } else if (scenario == "PRS+IDS+Cov+IDPs") {
    dat_cat <- cbind(dat1[, -c(1,2,6:11)], dat1_encoded)
    train_mat <- as.matrix(dat_cat)
  }
  cat("modality : ",scenario, "\n")
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
    prediction = TRUE,
    verbose=FALSE
  )
  
  predictions <- model_cv$pred
  roc_obj <- roc(dat1$Gla, predictions)
  auc_value <- pROC::auc(roc_obj)
  ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
  lower <- ci[1]
  upper <- ci[3]
  z <- 1.96
  se_est <- (upper - lower) / (2 * z)
  brier_score <- ModelMetrics::brier(actual = dat1$Gla, predicted = predictions)
  brier_individual <- (dat1$Gla - predictions)^2
  se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
  se_brier <- sqrt(var(brier_individual)/length(brier_individual))
  cat("AUC " , auc_value,"\n")
  cat("AUC DeLong SE:", round(se_est, 5), "\n")
  cat("Brier Score:", round(brier_score, 5), "\n")
  cat("Brier score SE:", round(se_brier, 5), "\n\n")
}


  
