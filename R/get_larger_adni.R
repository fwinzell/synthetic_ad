library(dplyr)
library(ggplot2)
library(ggpubr)
library(stats)
library(tidyr)

load_adni <- source("~/R/DDLS-R/load_adni.R")$value
adni_data <- load_adni()


ucsf_xsectional <- function() {
  # Load all Longitudinal UCSF datasets
  ucsf_data1 <- ucsffsx51final
  ucsf_data2 <- ucsffsx51
  
  cols <- intersect(colnames(ucsf_data1), colnames(ucsf_data2))
  ucsf_data1 <- ucsf_data1 %>% select(all_of(cols))
  ucsf_data2 <- ucsf_data2 %>% select(all_of(cols))
  
  ucsf_data <- rbind(ucsf_data1, ucsf_data2)
  
  ucsf_data3 <- ucsffsx
  cols <- intersect(colnames(ucsf_data), colnames(ucsf_data3))
  ucsf_data3 <- ucsf_data3 %>% select(all_of(cols))
  ucsf_data <- ucsf_data %>% select(all_of(cols))
  
  ucsf_data <- rbind(ucsf_data, ucsf_data3)
  
  ucsf_data4 <- ucsffsx6
  cols <- intersect(colnames(ucsf_data), colnames(ucsf_data4))
  ucsf_data4 <- ucsf_data4 %>% select(all_of(cols))
  ucsf_data <- ucsf_data %>% select(all_of(cols))
  
  ucsf_data <- rbind(ucsf_data, ucsf_data4)
  
  ucsf_data <- arrange(ucsf_data, RID)
  ucsf_data$RID <- as.numeric(ucsf_data$RID)
  
  var.names <- colnames(ucsf_data)[which(sub(".*([A-Z]{2})$", "\\1", colnames(ucsf_data)) %in% c("SV", "CV"))]
  ucsf_data <- select(ucsf_data, all_of(c("RID", "VISCODE", var.names)))
  
  return(list("data" = ucsf_data, "var.names" = var.names))
}




ucsf_data <- ucsf_xsectional()
mri_data <- ucsf_data[['data']] %>% filter(VISCODE %in% c('sc', 'bl', 'scmri')) %>% arrange(RID, VISCODE) %>%
  distinct(RID, .keep_all=TRUE)
log_mri_data <- mutate_all(mri_data[, ucsf_data[['var.names']]], log)

z_threshold <- 4
for (y_name in ucsf_data[['var.names']]) {
  z_scores <- scale(mri_data[,y_name])
  n_outliers <- sum(z_scores > z_threshold | z_scores < -z_threshold, na.rm = TRUE)
  
  z_log <- scale(log_mri_data[,y_name])
  n_log_outliers <- sum(z_log > z_threshold | z_log < -z_threshold, na.rm = TRUE)
  
  if (n_outliers > n_log_outliers) {
    z_scores <- z_log
    n_outliers <- n_log_outliers
  }
  print(paste("Outliers for", y_name, ": ", n_outliers))
  
  mri_data[which(z_scores > z_threshold | z_scores < -z_threshold), y_name] <- NA
  log_mri_data[which(z_scores > z_threshold | z_scores < -z_threshold), y_name] <- NA
}


adni_baseline <- adni_data[which(adni_data$VISCODE == "bl"), c("RID", "AGE", "PTGENDER", "PTEDUCAT",
                                                               "APOE4", "AV45", "ABETA", "TAU", "PTAU",
                                                               "ADAS13", "MMSE", "DX")]
# remove all rows with DX = NA
adni_baseline <- adni_baseline[!is.na(adni_baseline$DX),]
factor(adni_baseline$PTGENDER, levels = c("Male", "Female"), labels = c(1, 2)) -> adni_baseline$PTGENDER
factor(adni_baseline$DX, levels = c("CN", "MCI", "Dementia"), labels = c(0, 1, 2)) -> adni_baseline$DXN
adni_baseline$RID <- as.numeric(adni_baseline$RID)

adni_larger <- mri_data %>% dplyr::select(RID, ST130CV, ST129CV, ST127SV, ST121CV, ST119CV, 
                                   ST117CV, ST116CV, ST115CV, ST111CV, ST99CV, ST97CV, 
                                   ST96SV, ST91CV, ST90CV, ST89SV, ST88SV, ST85CV, ST83CV, 
                                   ST77SV, ST62CV, ST58CV, ST57CV, ST56CV, ST52CV, ST40CV, 
                                   ST38CV, ST37SV, ST32CV, ST31CV, ST30SV, ST29SV, ST26CV, 
                                   ST24CV, ST18SV) %>% full_join(adni_baseline, by="RID")
# split into test and train sets
smp_size <- floor(0.80 * nrow(adni_larger))

## set the seed to make your partition reproducible
set.seed(1)
train_ind <- sample(seq_len(nrow(adni_larger)), size = smp_size)

X_train <- adni_larger[train_ind, ]
X_test <- adni_larger[-train_ind, ]

write.csv(adni_larger, file = "~/R/data/DDLS/adni_plus.csv", row.names = FALSE)
write.csv(X_train, file = "~/R/data/DDLS/adni_plus_train.csv", row.names = FALSE)
write.csv(X_test, file = "~/R/data/DDLS/adni_plus_test.csv", row.names = FALSE)
