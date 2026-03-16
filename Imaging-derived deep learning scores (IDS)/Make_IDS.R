#####################################################################
##########                  Make IDS                       ##########
#####################################################################

library(data.table)
library(pROC)
library(dplyr)
library(xgboost)

file_name = "ConvNext_femto" #file_name is one of "ConvNext_femto", "PIT", "deit", "xcit", "VIT"

################### Making fold0.txt ##########################

folder_path <- paste0("/home/guestuser1/", file_name, "/fold/fold0/fold0_pred")
file_indices <- 0:127
file_paths <- file.path(folder_path, paste0(file_indices, "_pred.csv"))


df_list <- lapply(seq_along(file_paths), function(i) {
  path <- file_paths[i]
  df <- read.csv(path, stringsAsFactors = FALSE)
  df_summarised <- df %>%
    group_by(ID) %>%
    summarise(
      Actual = first(Actual),
      !!paste0("b_scan", i - 1) := mean(Predicted_Prob),
      .groups = "drop"
    )
  
  return(df_summarised)
})


merged_df <- Reduce(function(x, y) full_join(x, y, by = c("ID", "Actual")), df_list)


write.table(merged_df,
            file = paste0("/home/guestuser1/", file_name, "/fold/fold0/fold0.txt"),
            sep = "\t",
            row.names = FALSE,
            quote = FALSE)
nrow(merged_df)
##############################################################










################### Making fold1.txt ##########################
folder_path <- paste0("/home/guestuser1/", file_name, "/fold/fold1/fold1_pred")
file_indices <- 0:127
file_paths <- file.path(folder_path, paste0(file_indices, "_pred.csv"))

# 파일 처리
df_list <- lapply(seq_along(file_paths), function(i) {
  path <- file_paths[i]
  df <- read.csv(path, stringsAsFactors = FALSE)
  

  df_summarised <- df %>%
    group_by(ID) %>%
    summarise(
      Actual = first(Actual),
      !!paste0("b_scan", i - 1) := mean(Predicted_Prob),
      .groups = "drop"
    )
  
  return(df_summarised)
})

merged_df <- Reduce(function(x, y) full_join(x, y, by = c("ID", "Actual")), df_list)

write.table(merged_df,
            file = paste0("/home/guestuser1/", file_name, "/fold/fold1/fold1.txt"),
            sep = "\t",
            row.names = FALSE,
            quote = FALSE)
nrow(merged_df)
###############################################################


################### Making fold2.txt ##########################
folder_path <- paste0("/home/guestuser1/", file_name, "/fold/fold2/fold2_pred")
file_indices <- 0:127
file_paths <- file.path(folder_path, paste0(file_indices, "_pred.csv"))


df_list <- lapply(seq_along(file_paths), function(i) {
  path <- file_paths[i]
  df <- read.csv(path, stringsAsFactors = FALSE)
  

  df_summarised <- df %>%
    group_by(ID) %>%
    summarise(
      Actual = first(Actual),
      !!paste0("b_scan", i - 1) := mean(Predicted_Prob),
      .groups = "drop"
    )
  
  return(df_summarised)
})


merged_df <- Reduce(function(x, y) full_join(x, y, by = c("ID", "Actual")), df_list)


write.table(merged_df,
            file = paste0("/home/guestuser1/", file_name, "/fold/fold2/fold2.txt"),
            sep = "\t",
            row.names = FALSE,
            quote = FALSE)
nrow(merged_df)




################### Making fold3.txt ##########################
folder_path <- paste0("/home/guestuser1/", file_name, "/fold/fold3/fold3_pred")
file_indices <- 0:127
file_paths <- file.path(folder_path, paste0(file_indices, "_pred.csv"))


df_list <- lapply(seq_along(file_paths), function(i) {
  path <- file_paths[i]
  df <- read.csv(path, stringsAsFactors = FALSE)
  
 
  df_summarised <- df %>%
    group_by(ID) %>%
    summarise(
      Actual = first(Actual),
      !!paste0("b_scan", i - 1) := mean(Predicted_Prob),
      .groups = "drop"
    )
  
  return(df_summarised)
})


merged_df <- Reduce(function(x, y) full_join(x, y, by = c("ID", "Actual")), df_list)


write.table(merged_df,
            file = paste0("/home/guestuser1/", file_name, "/fold/fold3/fold3.txt"),
            sep = "\t",
            row.names = FALSE,
            quote = FALSE)
nrow(merged_df)
##############################################################



################### Making fold4.txt ##########################
folder_path <- paste0("/home/guestuser1/", file_name, "/fold/fold4/fold4_pred")
file_indices <- 0:127
file_paths <- file.path(folder_path, paste0(file_indices, "_pred.csv"))


df_list <- lapply(seq_along(file_paths), function(i) {
  path <- file_paths[i]
  df <- read.csv(path, stringsAsFactors = FALSE)
  

  df_summarised <- df %>%
    group_by(ID) %>%
    summarise(
      Actual = first(Actual),
      !!paste0("b_scan", i - 1) := mean(Predicted_Prob),
      .groups = "drop"
    )
  
  return(df_summarised)
})


merged_df <- Reduce(function(x, y) full_join(x, y, by = c("ID", "Actual")), df_list)


write.table(merged_df,
            file = paste0("/home/guestuser1/", file_name, "/fold/fold3/fold3.txt"),
            sep = "\t",
            row.names = FALSE,
            quote = FALSE)
nrow(merged_df)
##############################################################


#################### Merging fold.txt ########################
file_paths <- c(
  paste0("/home/guestuser1/", file_name, "/fold/fold0/fold0.txt"),
  paste0("/home/guestuser1/", file_name, "/fold/fold1/fold1.txt"),
  paste0("/home/guestuser1/", file_name, "/fold/fold2/fold2.txt"),
  paste0("/home/guestuser1/", file_name, "/fold/fold3/fold3.txt"),
  paste0("/home/guestuser1/", file_name, "/fold/fold4/fold4.txt")
)


df_list <- lapply(file_paths, function(path) {
  read.table(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
})


merged_all <- bind_rows(df_list)

write.table(merged_all,
            file = paste0("/home/guestuser1/", file_name, "/fold/fold_merged.txt"),
            sep = "\t",
            row.names = FALSE,
            quote = FALSE)



################ Making IDS(xgboost) ##############################
set.seed(123)


file_path <- paste0("/home/guestuser1/", file_name, "/fold/fold_merged.txt")
df <- read.table(file_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)

X <- as.matrix(df %>% select(starts_with("b_scan")))
y <- df$Actual
dtrain <- xgb.DMatrix(data = X, label = y)
set.seed(123)

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
result_df

write.table(result_df,
            file = paste0("/home/guestuser1/", file_name, "/IDS(xgboost).txt"),
            sep = "\t",
            row.names = FALSE,
            quote = FALSE)


#########################################################





