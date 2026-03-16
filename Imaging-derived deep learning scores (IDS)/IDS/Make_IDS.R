#####################################################################
##########                  Make IDS                       ##########
#####################################################################

library(data.table)
library(pROC)
library(dplyr)
library(xgboost)

file_name <- "ConvNext_femto"   # file_name is one of "ConvNext_femto", "PIT", "deit", "xcit", "VIT"
base_path <- paste0("/home/guestuser1/", file_name)

################### make fold ##########################
make_fold_file <- function(fold_num) {
  folder_path <- file.path(base_path, "fold", paste0("fold", fold_num), paste0("fold", fold_num, "_pred"))
  file_paths <- file.path(folder_path, paste0(0:127, "_pred.csv"))
  
  df_list <- lapply(seq_along(file_paths), function(i) {
    df <- read.csv(file_paths[i], stringsAsFactors = FALSE)
    
    df %>%
      group_by(ID) %>%
      summarise(
        Actual = first(Actual),
        !!paste0("b_scan", i - 1) := mean(Predicted_Prob),
        .groups = "drop"
      )
  })
  
  merged_df <- Reduce(function(x, y) full_join(x, y, by = c("ID", "Actual")), df_list)
  
  out_path <- file.path(base_path, "fold", paste0("fold", fold_num), paste0("fold", fold_num, ".txt"))
  write.table(merged_df, file = out_path, sep = "\t", row.names = FALSE, quote = FALSE)
  
  cat("fold", fold_num, "saved:", nrow(merged_df), "rows\n")
  return(merged_df)
}

################### Make fold0 ~ fold4 ##########################
fold_list <- lapply(0:4, make_fold_file)

#################### Merge all folds ########################
merged_all <- bind_rows(fold_list)

merged_path <- file.path(base_path, "fold", "fold_merged.txt")
write.table(merged_all, file = merged_path, sep = "\t", row.names = FALSE, quote = FALSE)

################ Making IDS (xgboost) ##############################
set.seed(123)

df <- read.table(merged_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)

X <- as.matrix(df %>% select(starts_with("b_scan")))
y <- df$Actual
dtrain <- xgb.DMatrix(data = X, label = y)

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
  prediction = TRUE
)

predictions <- model$pred

roc_obj <- roc(y, predictions)
auc_value <- auc(roc_obj)
print(paste("AUC:", auc_value))

result_df <- df %>%
  select(ID, Actual) %>%
  mutate(IDS = predictions)

write.table(
  result_df,
  file = file.path(base_path, "IDS(xgboost).txt"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
