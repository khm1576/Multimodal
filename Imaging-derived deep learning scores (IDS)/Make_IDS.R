#####################################################################
##########                  Make IDS                       ##########
#####################################################################

dat <- fread('/storage0/lab/khm1576/Workspace/disease/Glaucoma_dat.txt')
id <- fread('/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt')

dat_merge <- merge(dat,id[,-2],by.x = 'filter', by.y = 'V1')
dat_merge <- arrange(dat_merge, app)


for (fold_index in 0:4) {
  fold_folder <- paste0("fold", fold_index)
  setwd(paste0('/storage0/lab/khm1576/Image/OCT/', fold_folder, '/', fold_folder, '_pred'))
  files <- list.files(".", pattern = "_pred\\.csv")
  files <- files[order(as.numeric(gsub("_pred\\.csv", "", files)))]
  fold_file <- paste0('/storage0/lab/khm1576/IDPs/OCT/fold_', fold_index, '.csv')
  fold_data <- fread(fold_file)
  img <- dat_merge[app %in% fold_data$V1, c(1, 2)]
  
  for (file in files) {
    pr <- fread(file)
    column_name <- gsub("_pred\\.csv", "", file)
    result <- pr[, .(Actual = Actual[1], Predicted_Prob = mean(Predicted_Prob)), by = ID]
    setnames(result, "Predicted_Prob", column_name)
    result <- arrange(result, ID)
    img <- cbind(img, result[, 3, with = FALSE])
  }
  
  output_file <- paste0('/storage0/lab/khm1576/Image/OCT/', fold_folder, '/', fold_folder, '.txt')
  fwrite(img, output_file, row.names = FALSE, quote = FALSE, sep = "\t", col.names = TRUE)
}


b <- NULL
for (i in 0:4) {
  a <- fread(paste0('/storage0/lab/khm1576/Image/OCT/fold',i,'/fold',i,'.txt'))
  a[, (3:ncol(a)) := lapply(.SD, scale), .SDcols = 3:ncol(a)]
  b <- rbind(b,a)
}

b <- arrange(b,app14048)
fwrite(b,"/storage0/lab/khm1576/Image/OCT/merge_fold.txt",row.names=F,quote=F,sep="\t",col.names=T)

b <- cbind(b[,-c(1,2)],dat[(app14048 %in% id$V1),3])



dtrain <- xgb.DMatrix(data = as.matrix(b[,-129]), label = b$Gla)

set.seed(123)


# Cross-Validation
cv_result <- xgb.cv(
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
  verbose = 1,
  early_stopping_rounds = 10,
  prediction = T
)

hist(cv_result$pred)


roc_obj <- roc(b$Gla, cv_result$pred)
auc_value <- auc(roc_obj)
print(auc_value)


pred <- data.frame(app = id[,1],
                   IDS = cv_result$pred)
fwrite(pred,"/storage0/lab/khm1576/Image/OCT/IDS.txt",row.names=F,quote=F,sep="\t",col.names=T)
