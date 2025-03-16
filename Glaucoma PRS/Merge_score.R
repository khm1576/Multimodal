# Merge Score #

setwd('/storage0/lab/khm1576/ukb/pheno/Glaucoma/4.LDpred_score')


list <- c('_p1.0000e-01','_p1.0000e-02','_p1.0000e-03','_p1.0000e+00','_p3.0000e-01','_p3.0000e-02','_p3.0000e-03','-inf') 

for (ls in list) {
  res <- NULL
  for (i in 1:10) {
    fold <- fread(paste0("4.Glaucoma_fold",i,"_score_LDpred",ls,".txt"))
    fold$PRS_scale <- scale(fold$PRS)
    res <- rbind(res,fold)
  }
  fwrite(res, paste0('/storage0/lab/khm1576/ukb/pheno/Glaucoma/4.LDpred_score/merge/4.Glaucoma_score_LDpred_',ls,'.txt'), row.names=F, quote=F, sep="\t", col.names=T)
}
