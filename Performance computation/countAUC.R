library("pROC")
script_path <- dirname(sys.frame(1)$ofile)
result_list <- list()
AUC_data<- "countdata"
AUC_data_path <- file.path(script_path,AUC_data)
csv_files <- list.files(path = AUC_data_path, pattern = "\\.csv$", full.names = TRUE) 
for (csv_file in csv_files){
  df<-read.csv(csv_file)
  mycol <- c("slateblue","seagreen3","dodgerblue","firebrick1","lightgoldenrod","magenta","orange2")
  auc.out <- c()

  x <- plot.roc(df[,1],df[,2],ylim=c(0,1),xlim=c(1,0),
              smooth=F, 
              ci=TRUE, 
              main="",
              lwd=2, 
              legacy.axes=T)

  ci.lower <- round(as.numeric(x$ci[1]),3) 
  ci.upper <- round(as.numeric(x$ci[3]),3) 

  auc.ci <- c(colnames(df)[2],round(as.numeric(x$auc),3),paste(ci.lower,ci.upper,sep="-"))
  auc.out <- rbind(auc.out,auc.ci)



  for (i in 3:ncol(df)){
    x <- plot.roc(df[,1],df[,i],
                add=T, 
                smooth=F,
                ci=TRUE,
                col=mycol[i],
                lwd=2,
                legacy.axes=T)
  
    ci.lower <- round(as.numeric(x$ci[1]),3)
    ci.upper <- round(as.numeric(x$ci[3]),3)
    
    auc.ci <- c(colnames(df)[i],round(as.numeric(x$auc),3),paste(ci.lower,ci.upper,sep="-"))
    auc.out <- rbind(auc.out,auc.ci)
    
    
  }

  result_list[[basename(csv_file)]] <- auc.out
}
final_result <- do.call(rbind, result_list)


AUC_output_path <- "AUC_result.csv"
AUC_result_path <- file.path(script_path,AUC_output_path)

# Save the results as CSV file
write.csv(final_result, file = AUC_result_path, row.names = F)
