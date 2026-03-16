library(data.table)
library(pROC)
library(dplyr)
library(xgboost)
library(caret)
library(glmnet)
library(ModelMetrics)

# =========================================================
# 0. Common functions
# =========================================================

set.seed(123)

evaluate_glm_model <- function(data, outcome, features, model_name = "Model") {
  model_data <- data[, c(outcome, features), with = FALSE]
  model_data <- model_data[complete.cases(model_data)]

  x <- as.matrix(model_data[, ..features])
  y <- model_data[[outcome]]

  train_df <- data.frame(label = y, x)
  fit <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))

  pred <- as.numeric(predict(fit, type = "response"))

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

evaluate_glm_with_encoded_covariates <- function(data, outcome, numeric_features, factor_features, model_name = "Model") {
  model_data <- data[, c(outcome, numeric_features, factor_features), with = FALSE]
  model_data <- model_data[complete.cases(model_data)]

  encoded_factors <- model.matrix(~ . - 1, data = model_data[, ..factor_features])
  x <- cbind(
    as.matrix(model_data[, ..numeric_features]),
    encoded_factors
  )
  y <- model_data[[outcome]]

  train_df <- data.frame(label = y, x)
  fit <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))

  pred <- as.numeric(predict(fit, type = "response"))

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
  cat("Features:", paste(c(numeric_features, factor_features), collapse = ", "), "\n")
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
    features = c(numeric_features, factor_features),
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

# =========================================================
# 1. File paths and loading
# =========================================================

file_name <- "ConvNext_femto"   # one of: "ConvNext_femto", "PIT", "deit", "xcit", "VIT"
ids_path <- paste0("/home/guestuser1/", file_name, "/IDS/IDS(xgboost).txt")

cov_path <- "/storage0/lab/khm1576/연구주제/disease/Glaucoma_All_Cov.txt"
prs_path <- "/storage0/lab/khm1576/연구주제/PRS/Glaucoma_app14048.txt"
oct_path <- "/storage0/lab/khm1576/IDPs/OCT/OCT_IDPs.txt"
oct_id_path <- "/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt"
igs_path <- "/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS.txt"
igs2_path <- "/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS2.txt"

dat <- fread(cov_path)
prs <- fread(prs_path)
ids <- fread(ids_path)
oct <- fread(oct_path)
oct_id <- fread(oct_id_path)

# =========================================================
# 2. Preprocessing for 55k dataset
# =========================================================

oct <- oct[, !c("27851-0.0", "27853-0.0", "27855-0.0", "27857-0.0"), with = FALSE]
setnames(oct, old = names(oct), new = gsub("-0\\.0$", "", names(oct)))

setnames(ids, old = "ID", new = "app77890")

id_map <- unique(dat[, .(app77890, app14048)])
ids <- merge(ids, id_map, by = "app77890", all.x = TRUE)

dat_55k <- dat[(app14048 %in% oct_id$V1), -c("townsend"), with = FALSE]

ids_subset <- ids[, .(app14048, IDS)]
dat_55k <- merge(dat_55k, ids_subset, by = "app14048")

oct_sub <- oct[(app14048 %in% oct_id$V1)]
dat_55k <- cbind(dat_55k, oct_sub[, -1, with = FALSE])

dat_55k <- dat_55k[, -2, with = FALSE]

# =========================================================
# 3. Feature sets for 55k
# =========================================================

cov_numeric_55k <- c("age", "sex")
cov_factor_55k <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")

prs_feature <- "PRS_scale"
ids_feature <- "IDS"

base_cols_55k <- c("app14048", "Gla", cov_numeric_55k, cov_factor_55k, prs_feature, ids_feature)
idp_features_55k <- setdiff(names(dat_55k), base_cols_55k)

models_55k <- list(
  "PRS" = prs_feature,
  "PRS + IDPs" = c(prs_feature, idp_features_55k),
  "IDS" = ids_feature,
  "PRS + IDS" = c(prs_feature, ids_feature),
  "IDPs" = idp_features_55k,
  "PRS + IDS + Cov + IDPs" = c("PRS_scale", "IDS", "age", "sex", idp_features_55k)
)

# =========================================================
# 4. Run 55k models
# =========================================================

results_55k <- list()

results_55k[["Cov"]] <- evaluate_glm_with_encoded_covariates(
  data = dat_55k,
  outcome = "Gla",
  numeric_features = cov_numeric_55k,
  factor_features = cov_factor_55k,
  model_name = "[55k] Cov"
)

results_55k[["PRS"]] <- evaluate_glm_model(
  data = dat_55k,
  outcome = "Gla",
  features = models_55k[["PRS"]],
  model_name = "[55k] PRS"
)

results_55k[["PRS + IDPs"]] <- evaluate_glm_model(
  data = dat_55k,
  outcome = "Gla",
  features = models_55k[["PRS + IDPs"]],
  model_name = "[55k] PRS + IDPs"
)

results_55k[["IDS"]] <- evaluate_glm_model(
  data = dat_55k,
  outcome = "Gla",
  features = models_55k[["IDS"]],
  model_name = "[55k] IDS"
)

results_55k[["PRS + IDS"]] <- evaluate_glm_model(
  data = dat_55k,
  outcome = "Gla",
  features = models_55k[["PRS + IDS"]],
  model_name = "[55k] PRS + IDS"
)

results_55k[["IDPs"]] <- evaluate_glm_model(
  data = dat_55k,
  outcome = "Gla",
  features = models_55k[["IDPs"]],
  model_name = "[55k] IDPs"
)

results_55k[["PRS + IDS + Cov + IDPs"]] <- evaluate_glm_with_encoded_covariates(
  data = dat_55k[, c("Gla", "PRS_scale", "IDS", "age", "sex", cov_factor_55k, idp_features_55k), with = FALSE],
  outcome = "Gla",
  numeric_features = c("PRS_scale", "IDS", "age", "sex", idp_features_55k),
  factor_features = cov_factor_55k,
  model_name = "[55k] PRS + IDS + Cov + IDPs"
)

# =========================================================
# 5. Preprocessing for 400k dataset
# =========================================================

igs <- fread(igs_path)
igs2 <- fread(igs2_path)

setkey(igs, app14048)
setkey(igs2, app14048)
igs_merged <- merge(igs, igs2, by = "app14048", all = TRUE)

dat_400k <- cbind(dat[, -c("app77890", "townsend"), with = FALSE], igs_merged[, -1, with = FALSE])

dat_400k <- dat_400k[!(app14048 %in% oct_id$V1) & !is.na(Gla)]

cols_to_factor_400k <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")
dat_400k[, (cols_to_factor_400k) := lapply(.SD, as.factor), .SDcols = cols_to_factor_400k]

dat_400k_cov_complete <- dat_400k[
  complete.cases(dat_400k[, c("age", "sex", cols_to_factor_400k), with = FALSE])
]

# =========================================================
# 6. Feature sets for 400k
# =========================================================

cov_numeric_400k <- c("age", "sex")
cov_factor_400k <- cols_to_factor_400k

base_cols_400k <- c("app14048", "Gla", cov_numeric_400k, cov_factor_400k, prs_feature)
igs_features <- setdiff(names(dat_400k), base_cols_400k)

models_400k <- list(
  "PRS" = prs_feature,
  "IGSs" = igs_features,
  "PRS + IGSs" = c(prs_feature, igs_features),
  "PRS + IGSs + Cov" = c("PRS_scale", "age", "sex", igs_features)
)

# =========================================================
# 7. Run 400k models
# =========================================================

results_400k <- list()

results_400k[["Cov"]] <- evaluate_glm_with_encoded_covariates(
  data = dat_400k,
  outcome = "Gla",
  numeric_features = cov_numeric_400k,
  factor_features = cov_factor_400k,
  model_name = "[400k] Cov"
)

results_400k[["PRS"]] <- evaluate_glm_model(
  data = dat_400k,
  outcome = "Gla",
  features = models_400k[["PRS"]],
  model_name = "[400k] PRS"
)

results_400k[["IGSs"]] <- evaluate_glm_model(
  data = dat_400k,
  outcome = "Gla",
  features = models_400k[["IGSs"]],
  model_name = "[400k] IGSs"
)

results_400k[["PRS + IGSs"]] <- evaluate_glm_model(
  data = dat_400k,
  outcome = "Gla",
  features = models_400k[["PRS + IGSs"]],
  model_name = "[400k] PRS + IGSs"
)

results_400k[["PRS + IGSs + Cov"]] <- evaluate_glm_with_encoded_covariates(
  data = dat_400k[, c("Gla", "PRS_scale", "age", "sex", cov_factor_400k, igs_features), with = FALSE],
  outcome = "Gla",
  numeric_features = c("PRS_scale", "age", "sex", igs_features),
  factor_features = cov_factor_400k,
  model_name = "[400k] PRS + IGSs + Cov"
)

# =========================================================
# 8. Optional summary table
# =========================================================

summary_55k <- extract_summary(results_55k)
summary_400k <- extract_summary(results_400k)

cat("\n\n==================== 55k Summary ====================\n")
print(summary_55k)

cat("\n\n==================== 400k Summary ====================\n")
print(summary_400k)
