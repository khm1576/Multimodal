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


###55k

# Cov
colnames(dat1)
colnames(dat1[,c('age','sex','alcohol','smoke','illness','edu','ethnic','centre')])
set.seed(123)
lasso_model <- cv.glmnet(as.matrix(dat1[,c('age','sex','alcohol','smoke','illness','edu','ethnic','centre')]), dat1$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model, as.matrix(dat1[,c('age','sex','alcohol','smoke','illness','edu','ethnic','centre')]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat1$Gla, predicted_prob_lasso11)
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat1$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")



#####################################################################################

# PRS
#set.seed(123)
#lasso_model <- cv.glmnet(as.matrix(dat1[,c('PRS_scale')]), dat1$Gla, family = "binomial", alpha = 1)
#predicted_prob_lasso11 <- predict(lasso_model, as.matrix(dat1[,c('PRS_scale','IDS')]), type = "response", s = "lambda.min")
#roc_obj_lasso11 <- roc(dat1$Gla, predicted_prob_lasso11)
#auc(roc_obj_lasso11) 




#####################################################################################

# IDS
#lasso_model <- cv.glmnet(as.matrix(dat1[,c('IDS')]), dat1$Gla, family = "binomial", alpha = 1)
#predicted_prob_lasso11 <- predict(lasso_model, as.matrix(dat1[,c('PRS_scale','IDS')]), type = "response", s = "lambda.min")
#roc_obj_lasso11 <- roc(dat1$Gla, predicted_prob_lasso11)
#auc(roc_obj_lasso11) 


#####################################################################################
# PRS + IDPs
set.seed(123)
colnames(dat1)
lasso_model <- cv.glmnet(as.matrix(dat1[,c(3,13:54)]), dat1$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model, as.matrix(dat1[,c(3,13:54)]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat1$Gla, predicted_prob_lasso11)
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong  SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat1$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")


#####################################################################################

# PRS + IDS
set.seed(123)
lasso_model <- cv.glmnet(as.matrix(dat1[,c('PRS_scale','IDS')]), dat1$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model, as.matrix(dat1[,c('PRS_scale','IDS')]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat1$Gla, predicted_prob_lasso11)
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat1$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")

#####################################################################################

# IDPs
#dat1
colnames(dat1[,-c(1:12)])
set.seed(123)
lasso_model <- cv.glmnet(as.matrix(dat1[,-c(1:12)]), dat1$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model, as.matrix(dat1[,-c(1:12)]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat1$Gla, predicted_prob_lasso11)
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat1$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")


#####################################################################################

# PRS + IDS + Cov + IDPs
colnames(dat1)
#colnames(dat1[,-1])
colnames(dat1[,-c(1:2)])
set.seed(123)
lasso_model <- cv.glmnet(as.matrix(dat1[,-c(1:2)]), dat1$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model, as.matrix(dat1[,-c(1:2)]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat1$Gla, predicted_prob_lasso11)
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat1$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
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


# Cov
dat_cata <- dat1[complete.cases(dat1[, c("age", "sex", "alcohol", "smoke", "illness", "edu", "ethnic", "centre")])]
dat_cata[,c('age','sex',"alcohol", "smoke", "illness", "edu", "ethnic", "centre")]
lasso_model11 <- cv.glmnet(as.matrix(dat_cata[,c('age','sex',"alcohol", "smoke", "illness", "edu", "ethnic", "centre")]), dat_cata$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model11, as.matrix(dat_cata[,c('age','sex',"alcohol", "smoke", "illness", "edu", "ethnic", "centre")]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat_cata$Gla, predicted_prob_lasso11)
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat_cata$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")









# IGSs
colnames(dat1[,-c(1:11)])
dat1[,-c(1:11)]
lasso_model11 <- cv.glmnet(as.matrix(dat1[,-c(1:11)]), dat1$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model11, as.matrix(dat1[,-c(1:11)]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat1$Gla, predicted_prob_lasso11)
pROC::auc(roc_obj_lasso11) 
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat1$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")




# PRS + IGSs
colnames(dat1[,-c(1,2,4:11)])
lasso_model11 <- cv.glmnet(as.matrix(dat1[,-c(1,2,4:11)]), dat1$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model11, as.matrix(dat1[,-c(1,2,4:11)]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat1$Gla, predicted_prob_lasso11)
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat1$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")


# PRS + IGSs + Cov
dat_cata[,-c(1,2)]
lasso_model11 <- cv.glmnet(as.matrix(dat_cata[,-c(1,2)]), dat_cata$Gla, family = "binomial", alpha = 1)
predicted_prob_lasso11 <- predict(lasso_model11, as.matrix(dat_cata[,-c(1,2)]), type = "response", s = "lambda.min")
roc_obj_lasso11 <- pROC::roc(dat_cata$Gla, predicted_prob_lasso11)
pROC::auc(roc_obj_lasso11) 
ci <- pROC::ci.auc(roc_obj_lasso11,conf.level=0.95 ,method = "delong")
print(ci)
lower <- ci[1]
upper <- ci[3]
z <- 1.96
se_est <- (upper - lower) / (2 * z)
cat("DeLong SE:", round(se_est, 5), "\n")
auc_value <- pROC::auc(roc_obj_lasso11)
print(auc_value)

y <- dat_cata$Gla
p <- predicted_prob_lasso11
brier_score <- ModelMetrics::brier(actual = y, predicted = p)
brier_individual <- (p - y)^2
se_brier <- sd(brier_individual) / sqrt(length(brier_individual))
ci_lower <- brier_score - 1.96 * se_brier
ci_upper <- brier_score + 1.96 * se_brier
cat("Brier Score:", round(brier_score, 5), "\n")
cat("Standard Error (SE):", round(se_brier, 5), "\n")
cat("95% CI:", paste0("[", round(ci_lower, 5), ", ", round(ci_upper, 5), "]"), "\n")



