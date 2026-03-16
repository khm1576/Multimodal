library(data.table)
library(pROC)
library(dplyr)
library(glmnet)
library(ModelMetrics)

# =========================================================
# 0. Common functions
# =========================================================

set.seed(123)

evaluate_lasso_model <- function(data, outcome, features, model_name = "Model") {
  # use complete cases only
  model_data <- data[, c(outcome, features), with = FALSE]
  model_data <- model_data[complete.cases(model_data)]

  x <- as.matrix(model_data[, ..features])
  y <- model_data[[outcome]]

  fit <- cv.glmnet(
    x = x,
    y = y,
    family = "binomial",
    alpha = 1
  )

  pred <- as.numeric(
    predict(fit, newx = x, type = "response", s = "lambda.min")
  )

  roc_obj <- pROC::roc(y, pred)
  auc_ci <- pROC::ci.auc(roc_obj, conf.level = 0.95, method = "delong")
  auc_val <- as.numeric(pROC::auc(roc_obj))

  delong_se <- (auc_ci[3] - auc_ci[1]) / (2 * 1.96)

  brier_score <- ModelMetrics::brier(actual = y, predicted = pred)
  brier_individual <- (pred - y)^2
  brier_se <- sd(brier_individual) / sqrt(length(brier_individual))
  brier_ci_lower <- brier_score - 1.96 * brier_se
  brier_ci_upper <- brier_score + 1.96 * brier_se

  cat("\n=====================================================\n")
  cat("Model:", model_name, "\n")
  cat("Features:", paste(features, collapse = ", "), "\n")
  cat("N =", nrow(model_data), "\n")
  cat("AUC:", round(auc_val, 5), "\n")
  cat("95% CI (DeLong): [", round(auc_ci[1], 5), ", ", round(auc_ci[3], 5), "]\n", sep = "")
  cat("DeLong SE:", round(delong_se, 5), "\n")
  cat("Brier Score:", round(brier_score, 5), "\n")
  cat("Brier SE:", round(brier_se, 5), "\n")
  cat("95% CI (Brier): [", round(brier_ci_lower, 5), ", ", round(brier_ci_upper, 5), "]\n", sep = "")

  return(list(
    model_name = model_name,
    n = nrow(model_data),
    features = features,
    fit = fit,
    pred = pred,
    auc = auc_val,
    auc_ci = auc_ci,
    delong_se = delong_se,
    brier = brier_score,
    brier_se = brier_se,
    brier_ci = c(brier_ci_lower, brier_ci_upper)
  ))
}

# =========================================================
# 1. File paths and loading
# =========================================================

file_name <- "ConvNext_femto"   # one of: "ConvNext_femto", "PIT", "deit", "xcit", "VIT"
ids_path <- paste0("/home/guestuser1/", file_name, "/IDS/IDS(xgboost).txt")

cov_path  <- "/storage0/lab/khm1576/연구주제/disease/Glaucoma_All_Cov.txt"
prs_path  <- "/storage0/lab/khm1576/연구주제/PRS/Glaucoma_app14048.txt"
oct_path  <- "/storage0/lab/khm1576/IDPs/OCT/OCT_IDPs.txt"
oct_id_path <- "/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt"
igs_path  <- "/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS.txt"
igs2_path <- "/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS2.txt"

dat <- fread(cov_path)
prs <- fread(prs_path)
ids <- fread(ids_path)
oct <- fread(oct_path)
oct_id <- fread(oct_id_path)

# =========================================================
# 2. Preprocessing for 55k dataset
# =========================================================

# remove unnecessary OCT columns
oct <- oct[, !c("27851-0.0", "27853-0.0", "27855-0.0", "27857-0.0"), with = FALSE]
setnames(oct, old = names(oct), new = gsub("-0\\.0$", "", names(oct)))

# rename IDS ID column
setnames(ids, old = "ID", new = "app77890")

# app77890 -> app14048 mapping
id_map <- unique(dat[, .(app77890, app14048)])
ids <- merge(ids, id_map, by = "app77890", all.x = TRUE)

# keep only participants with OCT data
dat_55k <- dat[(app14048 %in% oct_id$V1), -c("townsend"), with = FALSE]

# merge IDS
ids_subset <- ids[, .(app14048, IDS)]
dat_55k <- merge(dat_55k, ids_subset, by = "app14048")

# merge OCT IDPs
oct_sub <- oct[(app14048 %in% oct_id$V1)]
dat_55k <- cbind(dat_55k, oct_sub[, -1, with = FALSE])

# keep original logic: remove second column
dat_55k <- dat_55k[, -2, with = FALSE]

# =========================================================
# 3. Feature sets for 55k
# =========================================================

cov_features <- c("age", "sex", "alcohol", "smoke", "illness", "edu", "ethnic", "centre")
prs_feature  <- "PRS_scale"
ids_feature  <- "IDS"

# replace previous index-based IDP selection with name-based selection
base_cols_55k <- c("app14048", "Gla", cov_features, prs_feature, ids_feature)
idp_features_55k <- setdiff(names(dat_55k), base_cols_55k)

# define models
models_55k <- list(
  "Cov" = cov_features,
  "PRS + IDPs" = c(prs_feature, idp_features_55k),
  "PRS + IDS" = c(prs_feature, ids_feature),
  "IDPs" = idp_features_55k,
  "PRS + IDS + Cov + IDPs" = c(cov_features, prs_feature, ids_feature, idp_features_55k)
)

# =========================================================
# 4. Run 55k models
# =========================================================

results_55k <- lapply(names(models_55k), function(model_name) {
  evaluate_lasso_model(
    data = dat_55k,
    outcome = "Gla",
    features = models_55k[[model_name]],
    model_name = paste0("[55k] ", model_name)
  )
})

names(results_55k) <- names(models_55k)

# =========================================================
# 5. Preprocessing for 400k dataset
# =========================================================

igs <- fread(igs_path)
igs2 <- fread(igs2_path)

setkey(igs, app14048)
setkey(igs2, app14048)
igs_merged <- merge(igs, igs2, by = "app14048", all = TRUE)

# combine dat and igs
dat_400k <- cbind(dat[, -c("app77890", "townsend"), with = FALSE], igs_merged[, -1, with = FALSE])

# exclude OCT participants and remove missing outcomes
dat_400k <- dat_400k[!(app14048 %in% oct_id$V1) & !is.na(Gla)]

# complete cases for covariates
dat_400k_cov_complete <- dat_400k[
  complete.cases(dat_400k[, ..cov_features])
]

# =========================================================
# 6. Feature sets for 400k
# =========================================================

base_cols_400k <- c("app14048", "Gla", cov_features, prs_feature)
igs_features <- setdiff(names(dat_400k), base_cols_400k)

models_400k <- list(
  "Cov" = cov_features,
  "IGSs" = igs_features,
  "PRS + IGSs" = c(prs_feature, igs_features),
  "PRS + IGSs + Cov" = c(cov_features, prs_feature, igs_features)
)

# select dataset for each model
data_for_400k_models <- list(
  "Cov" = dat_400k_cov_complete,
  "IGSs" = dat_400k,
  "PRS + IGSs" = dat_400k,
  "PRS + IGSs + Cov" = dat_400k_cov_complete
)

# =========================================================
# 7. Run 400k models
# =========================================================

results_400k <- lapply(names(models_400k), function(model_name) {
  evaluate_lasso_model(
    data = data_for_400k_models[[model_name]],
    outcome = "Gla",
    features = models_400k[[model_name]],
    model_name = paste0("[400k] ", model_name)
  )
})

names(results_400k) <- names(models_400k)

# =========================================================
# 8. Optional summary table
# =========================================================

extract_summary <- function(res_list) {
  rbindlist(lapply(res_list, function(x) {
    data.table(
      Model = x$model_name,
      N = x$n,
      AUC = round(x$auc, 5),
      AUC_CI_Lower = round(x$auc_ci[1], 5),
      AUC_CI_Upper = round(x$auc_ci[3], 5),
      DeLong_SE = round(x$delong_se, 5),
      Brier = round(x$brier, 5),
      Brier_CI_Lower = round(x$brier_ci[1], 5),
      Brier_CI_Upper = round(x$brier_ci[2], 5)
    )
  }))
}

summary_55k <- extract_summary(results_55k)
summary_400k <- extract_summary(results_400k)

cat("\n\n==================== 55k Summary ====================\n")
print(summary_55k)

cat("\n\n==================== 400k Summary ====================\n")
print(summary_400k)
