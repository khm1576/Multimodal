library(data.table)
library(pROC)
library(dplyr)
library(xgboost)
library(caret)
library(glmnet)
library(ModelMetrics)


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




#####################################################################################

# Cov
#dat1
factor_cols <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")
dat1_encoded <- model.matrix(~ . - 1, data = dat1[, ..factor_cols])
dat_cat <- cbind(dat1[,c(4,5)],dat1_encoded)
train_mat <- as.matrix(dat_cat)
print(colnames(train_mat))
#train_mat
colnames(train_mat)
label <- dat1$Gla
train_df <- data.frame(label = label, train_mat)
logit_model <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))



pred_prob <- predict(logit_model, type = "response")
roc_obj <- roc(train_df$label, pred_prob)
auc_value <- pROC::auc(roc_obj)
ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
print(ci)

# SE
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj)
print(auc_value)
#summary(logit_model)


brier_score <- brier(actual = train_df$label, predicted = pred_prob)
print(brier_score)
y <- train_df$labe
p <- pred_prob

# Brier Score 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2

# (SE)
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

# (CI, 95%)
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier


cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")

#####################################################################################

# PRS
#dat1
train_mat <- as.matrix(dat1[,c(3)])
#print(colnames(train_mat))
#train_mat
label <- dat1$Gla
train_df <- data.frame(label = label, train_mat)
colnames(train_df)
logit_model <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))


pred_prob <- predict(logit_model, type = "response")
roc_obj <- roc(train_df$label, pred_prob)
auc_value <- pROC::auc(roc_obj)
ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
print(ci)

# SE 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj)
print(auc_value)
#summary(logit_model)


brier_score <- brier(actual = train_df$label, predicted = pred_prob)
print(brier_score)
y <- train_df$labe
p <- pred_prob

# Brier Score
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2

# (SE)
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

# (CI, 95%)
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")
#####################################################################################
#PRS+IDPs
colnames(dat1)
train_mat <- as.matrix(dat1[,c(3,13:54)])
label <- dat1$Gla
train_df <- data.frame(label = label, train_mat)
colnames(train_df)
logit_model <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))


pred_prob <- predict(logit_model, type = "response")
roc_obj <- roc(train_df$label, pred_prob)
auc_value <- pROC::auc(roc_obj)
ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
print(ci)

# SE 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj)
print(auc_value)
#summary(logit_model)


brier_score <- brier(actual = train_df$label, predicted = pred_prob)
print(brier_score)
y <- train_df$labe
p <- pred_prob

# Brier Score 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2

# (SE)
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

# 
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")
#####################################################################################
# IDS
#dat1
train_mat <- as.matrix(dat1[,c(12)])
colnames(train_mat)
#print(colnames(train_mat))
#train_mat
label <- dat1$Gla
train_df <- data.frame(label = label, train_mat)
logit_model <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))

pred_prob <- predict(logit_model, type = "response")
roc_obj <- roc(train_df$label, pred_prob)
auc_value <- pROC::auc(roc_obj)
ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
print(ci)

# SE 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj)
print(auc_value)
#summary(logit_model)


brier_score <- brier(actual = train_df$label, predicted = pred_prob)
print(brier_score)
y <- train_df$labe
p <- pred_prob

# Brier Score 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2


se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")
#####################################################################################

# PRS + IDS
#dat1
train_mat <- as.matrix(dat1[,c(3,12)])
print(colnames(train_mat))
#train_mat
label <- dat1$Gla
train_df <- data.frame(label = label, train_mat)
logit_model <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))

pred_prob <- predict(logit_model, type = "response")
roc_obj <- roc(train_df$label, pred_prob)
auc_value <- pROC::auc(roc_obj)
ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
print(ci)

# SE 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj)
print(auc_value)
#summary(logit_model)


brier_score <- brier(actual = train_df$label, predicted = pred_prob)
print(brier_score)
y <- train_df$labe
p <- pred_prob

# Brier Score
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2


se_brier <- sd(brier_individual) / sqrt(length(brier_individual))


ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")

#####################################################################################

# IDPs
#dat1
train_mat <- as.matrix(dat1[,-c(1:12)])
colnames(train_mat)
#train_mat
label <- dat1$Gla
train_df <- data.frame(label = label, train_mat)
logit_model <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))

pred_prob <- predict(logit_model, type = "response")
roc_obj <- roc(train_df$label, pred_prob)
auc_value <- pROC::auc(roc_obj)
ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
print(ci)


lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj)
print(auc_value)
#summary(logit_model)


brier_score <- brier(actual = train_df$label, predicted = pred_prob)
print(brier_score)
y <- train_df$labe
p <- pred_prob

# Brier Score 계산 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2


se_brier <- sd(brier_individual) / sqrt(length(brier_individual))


ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")

#####################################################################################

# PRS + IDS + Cov + IDPs
dat_cat <- cbind(dat1[,-c(1,2,6:11)],dat1_encoded)
train_mat <- as.matrix(dat_cat)
colnames(train_mat)
label <- dat1$Gla
train_df <- data.frame(label = label, train_mat)
#colnames(train_df)
logit_model <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))

pred_prob <- predict(logit_model, type = "response")
roc_obj <- roc(train_df$label, pred_prob)
auc_value <- pROC::auc(roc_obj)
ci <- pROC::ci.auc(roc_obj,conf.level=0.95 ,method = "delong")
print(ci)

# 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj)
print(auc_value)
#summary(logit_model)


brier_score <- brier(actual = train_df$label, predicted = pred_prob)
print(brier_score)
y <- train_df$labe
p <- pred_prob

# Brier Score 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
# 각 샘플의 Brier 오차
brier_individual <- (p - y)^2

# 
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

#
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")




###############################400k###########################################

dat <- fread('/storage0/lab/khm1576/연구주제/disease/Glaucoma_All_Cov.txt')
igs <- fread('/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS.txt')
igs2 <- fread('/storage0/lab/khm1576/fastGWA_ex/LDpred/IGS2.txt')

colnames(igs)
nrow(igs)
ncol(igs)
igs
colnames(igs2)



setkey(igs, app14048)
setkey(igs2, app14048)
igs_merged <- merge(igs, igs2, by = "app14048", all = TRUE)
ncol(igs_merged)
igs<-igs_merged
ncol(igs)
colnames(igs)





dat1 <- cbind(dat[,-c('app77890','townsend')],igs[,-1])
id <- fread('/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt')
dat1 <- dat1[!(app14048 %in% id$V1)&!is.na(dat1$Gla),]
#dat1 <- dat1[(app14048 %in% id$V1)&!is.na(dat1$Gla),]
nrow(dat1)
ncol(dat1)


# cov

# factor
cols_to_factor <- c("alcohol", "smoke", "illness", "edu", "ethnic", "centre")

# 
dat1[, (cols_to_factor) := lapply(.SD, as.factor), .SDcols = cols_to_factor]
dat1[, lapply(.SD, function(x) sum(is.na(x))), .SDcols = cols_to_factor]
dat1
set.seed(123)
Gla_m2 <- glm(Gla ~ age + sex + alcohol + smoke + illness + edu + ethnic + centre, data = dat1, family = 'binomial')

yhat_test <- predict(Gla_m2,Gla_m2$model, type = 'response')
roc_obj1 <- pROC::roc(Gla_m2$model$Gla, yhat_test)
pROC::auc(roc_obj1)
ci <- pROC::ci.auc(roc_obj1,conf.level=0.95 ,method = "delong")
ci
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj1)
print(auc_value)


brier_score <- brier(actual = Gla_m2$model$Gla,predicted=yhat_test)
brier_score

y <- Gla_m2$model$Gla
p <- yhat_test

brier_score <- ModelMetrics::brier(actual = y, predicted = p)
#
brier_individual <- (p - y)^2

#
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

# (CI, 95%)
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")
###













# PRS
Gla_m1 <- glm(Gla ~ PRS_scale, data = dat1, family = 'binomial')
yhat_test <- predict(Gla_m1,dat1, type = 'response')

roc_obj1 <- pROC::roc(dat1$Gla, yhat_test)
pROC::auc(roc_obj1)

ci <- pROC::ci.auc(roc_obj1,conf.level=0.95 ,method = "delong")
print(ci)

# SE 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj1)
print(auc_value)

y <- dat1$Gla
p <- yhat_test

# Brier Score 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
# 
brier_individual <- (p - y)^2

# (SE)
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

# (CI, 95%)
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 출
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")












# IGS
nrow( dat1[,-c(1,3:11)])
colnames(dat1[,-c(1,3:11)])
Gla_m2 <- glm(Gla ~ ., data = dat1[,-c(1,3:11)], family = 'binomial')

yhat_test <- predict(Gla_m2,dat1[,-1], type = 'response')
roc_obj1 <- pROC::roc(dat1$Gla, yhat_test)
pROC::auc(roc_obj1)

ci <- pROC::ci.auc(roc_obj1,conf.level=0.95 ,method = "delong")
print(ci)

# SE 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj1)
print(auc_value)

y <- dat1$Gla
p <- yhat_test

# Brier Score 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
# 
brier_individual <- (p - y)^2

# 
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

# (CI, 95%)
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

#
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")

# PRS + IGSs
colnames(dat1[,-c(1,4:11)])
Gla_m2 <- glm(Gla ~ ., data = dat1[,-c(1,4:11)], family = 'binomial')

yhat_test <- predict(Gla_m2,dat1[,-1], type = 'response')
roc_obj1 <- pROC::roc(dat1$Gla, yhat_test)
pROC::auc(roc_obj1)

ci <- pROC::ci.auc(roc_obj1,conf.level=0.95 ,method = "delong")
print(ci)

# SE 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj1)
print(auc_value)

y <- dat1$Gla
p <- yhat_test

# Brier Score 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2

# (SE)
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

# (CI, 95%)
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")








# PRS + IGSs + Cov
colnames(dat1[,-c(1)])
Gla_m2 <- glm(Gla ~ ., data = dat1[,-c(1)], family = 'binomial')
yhat_test <- predict(Gla_m2,Gla_m2$model, type = 'response')
roc_obj1 <- pROC::roc(Gla_m2$model$Gla, yhat_test)
pROC::auc(roc_obj1)
ci <- pROC::ci.auc(roc_obj1,conf.level=0.95 ,method = "delong")
print(ci)

# SE 
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj1)
print(auc_value)

y <- Gla_m2$model$Gla
p <- yhat_test

# Brier Score 
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2

# (SE)
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))

# (CI, 95%)
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier

# 
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")












