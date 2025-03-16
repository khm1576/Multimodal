### LDpred ###

# step 1 #


rm(list=ls())

library(data.table)
library(glue)


for(fold in 1:10){
  print(glue(" #################### FOLD{fold} #################### "))
  print(glue('{Sys.time()}:: Start Load BOLT file:: fold{fold}'))
  Gla <- fread(glue("/storage0/lab/khm1576/ukb/pheno/Glaucoma/10fold/BOLT_UKB_Gla_fold{fold}.bgen.stats.gz"))
  print(glue('QC:: fold{fold}'))
  Gla <- Gla[Gla$A1FREQ > 0.01 & Gla$A1FREQ < 0.99,]
  
  print("Make file")
  Gla1 <- Gla2 <- Gla
  Gla1$SNPID <- paste0(Gla1$CHR,"_",Gla1$BP,"_",Gla1$ALLELE1,"_",Gla1$ALLELE0)
  Gla2$SNPID <- paste0(Gla2$CHR,"_",Gla2$BP,"_",Gla2$ALLELE0,"_",Gla2$ALLELE1)
  
  print(glue('{Sys.time()}:: Start Load Bim file:: fold{fold}'))
  bim <- fread(glue("/storage0/dataUKB/UKB10fold/459188/Fold{fold}_1.2M.bim"))
  bim$SNP <- paste0(bim$V1,"_",bim$V4,"_",bim$V5,"_",bim$V6)
  
  Gla1 <- merge(Gla1,bim[,c("V2","SNP")],by.x="SNPID",by.y="SNP",sort=F)
  Gla2 <- merge(Gla2,bim[,c("V2","SNP")],by.x="SNPID",by.y="SNP",sort=F)
  
  Gla <- rbind(Gla1,Gla2);rm(Gla1,Gla2)
  
  
  fwrite(Gla,glue("/storage0/lab/khm1576/ukb/pheno/Glaucoma/0.Glaucoma_fold{fold}_gwas.txt"),row.names=F,quote=F,sep="\t",col.names=T)
  
  res <- data.table(hg19chrc=paste0("chr",Gla$CHR),
                    snpid=Gla$V2,
                    a1=Gla$ALLELE1,
                    a2=Gla$ALLELE0,
                    bp=Gla$BP,
                    or=exp(Gla$BETA),
                    p=Gla$P_BOLT_LMM)
  print(paste0("The number of duplicated SNPs in reference and summary data=",length(intersect(res$snpid,bim$V2))))
  print("Save")
  
  print(glue('{Sys.time()}:: Start Save txt file:: fold{fold}'))
  fwrite(res,glue("/storage0/lab/khm1576/ukb/pheno/Glaucoma/1.bolt_summary/1.Glaucoma_sumstat_fold{fold}.txt"),row.names=F,quote=F,sep="\t",col.names=T)
  print("Finish")
  print(glue('{Sys.time()}:: End Save txt file:: fold{fold}'))
  
}
