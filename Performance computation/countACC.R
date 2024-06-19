library(Hmisc)
script_path <- dirname(sys.frame(1)$ofile)
ACC_data <- "countdata"
ACC_data_path <- file.path(script_path,ACC_data)
csv_files <- list.files(path = ACC_data_path, pattern = "\\.csv$", full.names = TRUE)  # Get all CSV files in this path

result_list <- list()
cutoff_data <- "model cutoff.csv"  # "Imported optimal cutoff value table"
cutoff_data_path<-file.path(script_path,cutoff_data)
cut_off_data <- read.csv(cutoff_data_path)

for (csv_file in csv_files) {
  data <- read.csv(csv_file)
  data1 <- data
  data2 <- data
  data3 <- data.frame()
  
  for (j in 2:ncol(data)) {
    cut_off <- cut_off_data[1, j]
    for (i in 1:nrow(data)) {
      value1 <- data[i, j]
      label <- data[i, 1]
      
      if (value1 >= cut_off) {
        value1 <- 1
      } else {
        value1 <- 0
      }
      
      if (label == 1) {
        if (value1 == 1) {
          value2 <- 'TP'
        } else {
          value2 <- 'FN'
        }
      } else {
        if (value1 == 0) {
          value2 <- 'TN'
        } else {
          value2 <- 'FP'
        }
      }
      
      data1[i, j] <- value1
      data2[i, j] <- value2
    }
  }
  
  column_names <- colnames(data)
  column_names <- column_names[-1]
  num1 <- length(column_names)
  target_value1 <- "TP"
  target_value2 <- "TN"
  target_value3 <- "FP"
  target_value4 <- "FN"
  
  for (n in 1:num1) {
    count_target_TP <- sum(data2[, n + 1] == target_value1)
    count_target_TN <- sum(data2[, n + 1] == target_value2)
    count_target_FP <- sum(data2[, n + 1] == target_value3)
    count_target_FN <- sum(data2[, n + 1] == target_value4)
    data3[n, 1] <- count_target_TP
    data3[n, 2] <- count_target_TN
    data3[n, 3] <- count_target_FP
    data3[n, 4] <- count_target_FN
    
    ci.Accuracy <-round( binconf(count_target_TP + count_target_TN, count_target_TP + count_target_TN + count_target_FN + count_target_FP),3)
    ci.Sensitivity <- round(binconf(count_target_TP ,count_target_TP + count_target_FN),3)
    ci.Specificity <- round(binconf(count_target_TN ,count_target_TN + count_target_FP),3)
    ci.PPV <- round(binconf(count_target_TP , count_target_TP + count_target_FP),3)
    ci.NPV <- round(binconf(count_target_TN ,count_target_TN + count_target_FN),3)
    
    
    data3[n, 5] <-  paste(ci.Accuracy[1], "(", ci.Accuracy[2], "~", ci.Accuracy[3], ")", sep = "")
    data3[n, 6] <-  paste(ci.Sensitivity[1], "(", ci.Sensitivity[2], "~", ci.Sensitivity[3], ")", sep = "")
    data3[n, 7] <-  paste(ci.Specificity[1], "(", ci.Specificity[2], "~", ci.Specificity[3], ")", sep = "")
    data3[n, 8] <-  paste(ci.PPV[1], "(", ci.PPV[2], "~", ci.PPV[3], ")", sep = "")
    data3[n, 9] <-  paste(ci.NPV[1], "(", ci.NPV[2], "~", ci.NPV[3], ")", sep = "")
    
    rownames(data3)[n] <- column_names[n]
    colnames(data3)[1] <- 'TP'
    colnames(data3)[2] <- 'TN'
    colnames(data3)[3] <- 'FP'
    colnames(data3)[4] <- 'FN'
    colnames(data3)[5] <- 'Accuracy'
    colnames(data3)[6] <- 'Sensitivity'
    colnames(data3)[7] <- 'Specificity'
    colnames(data3)[8] <- 'PPV'
    colnames(data3)[9] <- 'NPV'
  }
  result_list[[basename(csv_file)]] <- data3
}



final_result <- do.call(rbind, result_list)


ACC_output_path <- "ACC_result.csv"
ACC_result_path <- file.path(script_path,ACC_output_path)

# Save the results as CSV file
write.csv(final_result, file = ACC_result_path, row.names = TRUE)
