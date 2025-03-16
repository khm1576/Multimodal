#####################################################################
##########                   Plink score                   ##########
#####################################################################

for (folder in folders) {
  setwd(paste0('/storage0/lab/khm1576/fastGWA_ex/LDpred/',folder))
  files <- list.files(".","2.gi")
  files <- files[-grep("log",files)]
  
  for(file in files){
    out <- gsub("2.gibbs","3.gibbs",file,fixed=T)
    out <- gsub(".txt","",out,fixed=T)
    
    plink <- glue("/storage0/program/plink_linux_x86_64_20200616/plink1.9 --bfile /storage0/lab/khm1576/ukb/pheno/Glaucoma/10fold/ALL_1.2M --bim /storage0/lab/khm1576/fastGWA_ex/ALL_1.2M.bim --score {file} 3 4 7 sum --out {out}")
    sbatch <- glue("/usr/bin/sbatch --nodelist compute-0-0 --mem=30G -o {out}.log -J score --wrap='{plink}'")
    system(sbatch)
  }
}
