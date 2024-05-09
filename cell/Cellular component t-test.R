library(data.table)
library(dplyr)
library(readxl)
library(dplyr)
library(writexl)
library(stringr)
library(broom)
importance <- c(paste0("model ranking")) 
topnum<-c(11)#topnum can be set to output inspection results within different ranges.
data <- read_excel("C:/Users/32618/Desktop/cell/cell data/iSCLM.xlsx", sheet = 1, col_types = c("text", "text", "numeric", "numeric", "numeric", "text", "text", "text", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"))
#data1 Using the provided label list address.
data1 <- read_excel("C:/Users/32618/Desktop/cell/cell data/label list.xlsx", sheet = 1, col_types = c("numeric", "text"))
for (importancename in importance) {
  
  data2 <- filter(data, get(importancename)<topnum) 
  
  grouped_data_top10 <- data2 %>%
    group_by(`Patient number`)

  summary_table_top10 <- grouped_data_top10 %>%
    summarise(
      total_no.neo = sum(no.neo),
      total_neopla = sum(neopla),
      total_nolabe = sum(nolabe),
      total_connec = sum(connec),
      total_inflam = sum(inflam),
      total_necros = sum(necros),
      total_all = sum(all),
      patch_count = n(),
      Tumor_count = sum(str_count(type, "tumor")),
      Stroma_count = sum(str_count(type, "stroma"))
    )


  summary_table_top10 <- merge(summary_table_top10, data1, by.x = "Patient number", by.y = "Patient number", all.x = TRUE, all.y = FALSE)

  summary_table_top10 <- summary_table_top10 %>%
    mutate(
      density_no.neo = round(total_no.neo / patch_count, 5),
      density_neopla = round(total_neopla / patch_count, 5),
      density_nolabe = round(total_nolabe / patch_count, 5),
      density_connec = round(total_connec / patch_count, 5),
      density_inflam = round(total_inflam / patch_count, 5),
      density_necros = round(total_necros / patch_count, 5),
      density_all = round(total_all / patch_count, 5)
    )

  # Calculate the proportion of cells: divide each column by the total number of cells.
  summary_table_top10 <- summary_table_top10 %>%
    mutate(
      proportion_no.neo = (round((total_no.neo / total_all) * 100, 5)),
      proportion_neopla = (round((total_neopla / total_all) * 100, 5)),
      proportion_nolabe = (round((total_nolabe / total_all) * 100, 5)),
      proportion_connec = (round((total_connec / total_all) * 100, 5)),
      proportion_inflam = (round((total_inflam / total_all) * 100, 5)),
      proportion_necros = (round((total_necros / total_all) * 100, 5)),
      proportion_all = (round((total_all / total_all) * 100, 5))
    )

  # Calculate the cell proportion: divide each type of cells by the total number of cells in neolpa.
  summary_table_top10 <- summary_table_top10 %>%
    mutate(
      ratio_no.neo = (round((total_no.neo / total_neopla) * 100, 5)),
      ratio_neopla = (round((total_neopla / total_neopla) * 100, 5)),
      ratio_nolabe = (round((total_nolabe / total_neopla) * 100, 5)),
      ratio_connec = (round((total_connec / total_neopla) * 100, 5)),
      ratio_inflam = (round((total_inflam / total_neopla) * 100, 5)),
      ratio_necros = (round((total_necros / total_neopla) * 100, 5))
    )
  #---------------------------------------whole roi------------------------------------#
  grouped_data <- data %>%
    group_by(`Patient number`)
  
  summary_table <- grouped_data %>%
    summarise(
      total_no.neo = sum(no.neo),
      total_neopla = sum(neopla),
      total_nolabe = sum(nolabe),
      total_connec = sum(connec),
      total_inflam = sum(inflam),
      total_necros = sum(necros),
      total_all = sum(all),
      patch_count = n(),
      Tumor_count = sum(str_count(type, "tumor")),
      Stroma_count = sum(str_count(type, "stroma"))
    )
  
  
  summary_table <- merge(summary_table, data1, by.x = "Patient number", by.y = "Patient number", all.x = TRUE, all.y = FALSE)
  
  summary_table<- summary_table %>%
    mutate(
      density_no.neo = round(total_no.neo / patch_count, 5),
      density_neopla = round(total_neopla / patch_count, 5),
      density_nolabe = round(total_nolabe / patch_count, 5),
      density_connec = round(total_connec / patch_count, 5),
      density_inflam = round(total_inflam / patch_count, 5),
      density_necros = round(total_necros / patch_count, 5),
      density_all = round(total_all / patch_count, 5)
    )
  
  # Calculate cell proportions: Divide each column by the total number of cells.
  summary_table <- summary_table %>%
    mutate(
      proportion_no.neo = (round((total_no.neo / total_all) * 100, 5)),
      proportion_neopla = (round((total_neopla / total_all) * 100, 5)),
      proportion_nolabe = (round((total_nolabe / total_all) * 100, 5)),
      proportion_connec = (round((total_connec / total_all) * 100, 5)),
      proportion_inflam = (round((total_inflam / total_all) * 100, 5)),
      proportion_necros = (round((total_necros / total_all) * 100, 5)),
      proportion_all = (round((total_all / total_all) * 100, 5))
    )
  
  # Calculate cell ratio: Divide the class cells by the total number of cells in neolpa.
  summary_table <- summary_table %>%
    mutate(
      ratio_no.neo = (round((total_no.neo / total_neopla) * 100, 5)),
      ratio_neopla = (round((total_neopla / total_neopla) * 100, 5)),
      ratio_nolabe = (round((total_nolabe / total_neopla) * 100, 5)),
      ratio_connec = (round((total_connec / total_neopla) * 100, 5)),
      ratio_inflam = (round((total_inflam / total_neopla) * 100, 5)),
      ratio_necros = (round((total_necros / total_neopla) * 100, 5))
    )

  # Save the raw data
  write_xlsx(summary_table_top10, paste0("C:/Users/32618/Desktop/cell/cell sort/Patient unit cell top ", topnum - 1, "_", importancename, " data.xlsx"))
  write_xlsx(summary_table, paste0("C:/Users/32618/Desktop/cell/cell sort/Patient unit cell whoel ROI ", importancename, " data.xlsx"))

  # Independent samples t-test.
  data3 <- read_excel(paste0("C:/Users/32618/Desktop/cell/cell sort/Patient unit cell top ", topnum - 1, "_", importancename, " data.xlsx"))
  data4 <- read_excel(paste0("C:/Users/32618/Desktop/cell/cell sort/Patient unit cell whoel ROI ", importancename, " data.xlsx"))
  data3 <- data3 %>% filter(`Label` %in% c("0", "1"))
  data4 <- data4 %>% filter(`Label` %in% c("0", "1"))
  columns <- c("proportion_no.neo","proportion_neopla","proportion_nolabe", "proportion_connec", "proportion_inflam", "proportion_necros","ratio_no.neo", "ratio_nolabe", "ratio_connec", "ratio_inflam", "ratio_necros")
  results1 <- list()
  results2 <- list()
  for (col in columns) {
    res1 <- t.test(
      data3[data3$Label == "0", ][[col]],
      data3[data3$Label == "1", ][[col]]
    )
    results1[[col]] <- tidy(res1)
    
    res2 <- t.test(
      data4[data4$Label == "0", ][[col]],
      data4[data4$Label == "1", ][[col]]
    )
    results2[[col]] <- tidy(res2)
  }
  results_df1 <- bind_rows(results1, .id = "variable")
  results_df2 <- bind_rows(results2, .id = "variable")
  # Save the t-test results
  write_xlsx(results_df1, paste0("C:/Users/32618/Desktop/cell/result/Patient unit cell top ", topnum - 1, "_", importancename, " t-test.xlsx"))
  write_xlsx(results_df2, paste0("C:/Users/32618/Desktop/cell/result/Patient unit cell whoel ROI ", importancename, " t-test.xlsx"))
}
