library(xlsx)
# Set folder for storing top patch distance mean data 
folder_path <- "C:/Users/32618/Desktop/distance(table S5)/Top patch average distance"
xlsx_files <- list.files(folder_path, pattern = "\\.xlsx$", full.names = TRUE)
result_list <- list()

for (xlsx_file in xlsx_files) {
  data <- read.xlsx(xlsx_file, 1)
  data_result <- data.frame()
  Label_column <- data[, 12]
  features <- data[, -c(1, 12:15)]
  file_name <- tools::file_path_sans_ext(basename(xlsx_file))
  for (col in 1:ncol(features)) {
    features_vector <- as.numeric(features[, col])
    # Perform Wilcoxon test and extract p-values
    p_value <- wilcox.test(Label_column, features_vector, paired = TRUE)$p.value
    if (p_value < 0.05) {
      data_result[col, 1] <- paste("P-value for  ")
      data_result[col, 2] <- paste("top", col * 5, "distance from the mean and overall mean distance test")
      data_result[col, 3] <- paste(round(p_value, 3))
      data_result[col, 4] <- paste("Statistically significant")
    } else {
      data_result[col, 1] <- paste("P-value for  ")
      data_result[col, 2] <- paste("top", col * 5, "distance from the mean and overall mean distance test")
      data_result[col, 3] <- paste(round(p_value, 3))
      data_result[col, 4] <- paste("Not statistically significant")
    }
    print(data_result[col, 1])
  }
  result_list[[basename(file_name)]] <- data_result
}
# Merge calculation results
final_result <- do.call(rbind, result_list)
output_path <- "C:/Users/32618/Desktop/distance(table S5)/result.xlsx"
write.xlsx(final_result, file = output_path, row.names = T ,col.names = F)
