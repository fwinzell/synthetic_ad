library(dplyr)
library(ggplot2)
library(ggpubr)
library(stats)
library(tidyr)

load_adni <- source("~/R/DDLS-R/load_adni.R")$value
#merge_adni <- source("~/R/DDLS-R/merge_adni.R")$value
getLongDX <- source("~/R/DDLS-R/getLongDX.R")$value
adni_data <- load_adni()

# ---- MRI outliers ----
mrivars <- c("Ventricles", "Hippocampus", "WholeBrain", 
             "Entorhinal", "Fusiform", "MidTemp", "ICV")
mri_data <- adni_data[, mrivars]
log_mri_data <- mutate_all(mri_data, log)

z_threshold <- 4
for (y_name in mrivars) {
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

adni_data[, mrivars] <- mri_data

###############################
####### Baseline data #########
###############################

adni_baseline <- adni_data[which(adni_data$VISCODE == "bl"), c("RID", "AGE", "PTGENDER", "PTEDUCAT",
                                                               "APOE4", "AV45", "ABETA", "TAU", "PTAU",
                                                               "ADAS13", "MMSE", 
                                                               "Ventricles", "Hippocampus", "WholeBrain", 
                                                               "Entorhinal", "Fusiform", "MidTemp", "ICV", "DX")]
# remove all rows with DX = NA
adni_baseline <- adni_baseline[!is.na(adni_baseline$DX),]
factor(adni_baseline$PTGENDER, levels = c("Male", "Female"), labels = c(1, 2)) -> adni_baseline$PTGENDER
factor(adni_baseline$DX, levels = c("CN", "MCI", "Dementia"), labels = c(0, 1, 2)) -> adni_baseline$DXN

# split into test and train sets
smp_size <- floor(0.80 * nrow(adni_baseline))

## set the seed to make your partition reproducible
set.seed(1)
train_ind <- sample(seq_len(nrow(adni_baseline)), size = smp_size)

X_train <- adni_baseline[train_ind, ]
X_test <- adni_baseline[-train_ind, ]

write.csv(adni_baseline, file = "~/R/data/DDLS/adni_bl_outliers_removed.csv", row.names = FALSE)
write.csv(X_train, file = "~/R/data/DDLS/adni_train.csv", row.names = FALSE)
write.csv(X_test, file = "~/R/data/DDLS/adni_test.csv", row.names = FALSE)

######## New DX labels ##########
# Transition vs Stable
adni_baseline_2 <- adni_baseline |> filter(DX != "Dementia") |> getLongDX(adni_data)
factor(adni_baseline_2$DX, 
       levels = c( "CN stable", "MCI stable", "Transition CN to MCI", "Transition MCI to Dementia", "Transition CN to Dementia" ), 
       labels = c(0, 1, 2, 3, 4)) -> adni_baseline_2$DXN

train_ind <- sample(seq_len(nrow(adni_baseline_2)), size = floor(0.80 * nrow(adni_baseline_2)))
X_train_2 <- adni_baseline_2[train_ind, ]
X_test_2 <- adni_baseline_2[-train_ind, ]

write.csv(adni_baseline_2, file = "~/R/data/DDLS/adni_bl_outliers_removed_2.csv", row.names = FALSE)
write.csv(X_train_2, file = "~/R/data/DDLS/adni_train_2.csv", row.names = FALSE)
write.csv(X_test_2, file = "~/R/data/DDLS/adni_test_2.csv", row.names = FALSE)

# New labels
# AB+ vs AB-
adni_baseline_ab <- adni_baseline |> filter(DX != "Dementia")
adni_baseline_ab <- adni_baseline_ab |> mutate(AB = ifelse(AV45 < 1.11, 0, 1))

###################################
###### Longitudinal data ##########
###################################

adni_long <- adni_data[, c("RID", "Month", "AGE", "PTGENDER", "PTEDUCAT",
                                      "APOE4", "AV45", "ABETA", "TAU", "PTAU",
                                      "ADAS13", "MMSE", 
                                    "Ventricles", "Hippocampus", "WholeBrain", 
                            "Entorhinal", "Fusiform", "MidTemp", "ICV", "DX")]
adni_long <- adni_long[!is.na(adni_long$DX),]
factor(adni_long$PTGENDER, levels = c("Male", "Female"), labels = c(1, 2)) -> adni_long$PTGENDER
factor(adni_long$DX, levels = c("CN", "MCI", "Dementia"), labels = c(0, 1, 2)) -> adni_long$DXN

X_train_long <- adni_long[adni_long$RID %in% X_train$RID, ]
X_test_long <- adni_long[adni_long$RID %in% X_test$RID, ]

write.csv(adni_long, file = "~/R/data/DDLS/adni_long_outliers_removed.csv", row.names = FALSE)
write.csv(X_train_long, file = "~/R/data/DDLS/adni_train_long.csv", row.names = FALSE)
write.csv(X_test_long, file = "~/R/data/DDLS/adni_test_long.csv", row.names = FALSE)






