#####################################################################
##########                  Make sumstat                   ##########
#####################################################################

rm(list=ls())

library(data.table)
library(glue)

field <- c(27801,27822,27823,28500,28501,28502,28503,28504,28505,28506,28507,28508,28509,28510,28511,28512,28513,28514,28515,28516,28517,28518,28519,28520,28521,28522,28523,28524,28525,28526,28527,28528,28529,28530,28531,28532,28533,28534,28535,28536,28537)

bim <- fread(glue("/storage0/lab/khm1576/ukb/pheno/Glaucoma/10fold/ALL_1.2M.bim"))
bim$SNP <- paste0(bim$V1,"_",bim$V4,"_",bim$V5,"_",bim$V6)

for (i in field) {
  
  OCT <- fread(paste0('/storage0/lab/khm1576/fastGWA_ex/OCT/',i,'_assoc.fastGWA'))
  
  OCT <- OCT[OCT$AF1 > 0.01 & OCT$AF1 < 0.99,]
  
  OCT1 <- OCT2 <- OCT
  OCT1$SNPID <- paste0(OCT1$CHR,"_",OCT1$POS,"_",OCT1$A1,"_",OCT1$A2)
  OCT2$SNPID <- paste0(OCT2$CHR,"_",OCT2$POS,"_",OCT2$A2,"_",OCT2$A1)
  
  OCT1 <- merge(OCT1,bim[,c("V2","SNP")],by.x="SNPID",by.y="SNP",sort=F)
  OCT2 <- merge(OCT2,bim[,c("V2","SNP")],by.x="SNPID",by.y="SNP",sort=F)
  
  OCT <- rbind(OCT1,OCT2);rm(OCT1,OCT2)
  
  fwrite(OCT,paste0("/storage0/lab/khm1576/fastGWA_ex/LDpred/",i,"/0.Full.txt"),row.names=F,quote=F,sep="\t",col.names=T)
  
  res <- data.table(hg19chrc=paste0("chr",OCT$CHR),
                    snpid=OCT$V2,
                    a1=OCT$A1,a2=OCT$A2,bp=OCT$POS,
                    or=exp(OCT$BETA),p=OCT$P)
  
  fwrite(res,paste0("/storage0/lab/khm1576/fastGWA_ex/LDpred/",i,"/1.sumstat.txt"),row.names=F,quote=F,sep="\t",col.names=T)
  
}
