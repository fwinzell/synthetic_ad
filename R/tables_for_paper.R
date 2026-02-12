train_ds <- read.csv("~/R/data/DDLS/adni_train.csv")
train_ml <- read.csv("~/R/data/DDLS/adni_train_ml.csv")
test <- read.csv("~/R/data/DDLS/adni_test.csv")

all_data <- rbind(train_ml, test)

synthetic <- read.csv("~/WASP-DDLS/DS-synthetic-data/degree3_deter/bn_adni_AGE.csv")

# Missing values
colMeans(is.na(train_ds))
colMeans(is.na(train_ml))
colMeans(is.na(adni_test))
colMeans(is.na(synthetic))

(colSums(is.na(train_ds)) - colSums(is.na(synthetic)))/nrow(train_ds)


# Analysis of data

plot_histogram <- function(data, column_name) {
  ks <- ks.test(data[,column_name], "pnorm")
  hist <- ggplot(data, aes_string(x = column_name)) +
    geom_histogram(bins = round(sqrt(length(data[,column_name]))),
                   fill = "steelblue", color = "black") +
    labs(title = paste("Histogram of", column_name, "KS=", ks$statistic), x = column_name, y = "Frequency")
  
  return(hist)
}


# MRI
# ---- MRI outliers ----
mrivars <- c("Ventricles", "Hippocampus", "WholeBrain", 
             "Entorhinal", "Fusiform", "MidTemp", "ICV")
mri_data <- all_data[, mrivars]
log_mri_data <- mutate_all(mri_data, log)


for (y_name in mrivars) {
  hist <- plot_histogram(mri_data, y_name)
  loghist <- plot_histogram(log_mri_data, y_name)
  
  print(ggarrange(hist, loghist, ncol = 2))
}

z_thresh <- 4
for (y_name in mrivars) {
  z_scores <- na.omit(scale(mri_data[,y_name]))
  n_outliers <- sum(z_scores > z_thresh | z_scores < -z_thresh)
  print(paste("Outliers for", y_name, " : ", n_outliers))
  
  z_log <- na.omit(scale(log_mri_data[,y_name]))
  n_log_outliers <- sum(z_log > z_thresh | z_log < -z_thresh)
  print(paste("Outliers for log(", y_name, ") : ", n_log_outliers))
  
  zhist <- ggplot(data.frame(z = z_scores, z_log = z_log)) +
    geom_histogram(aes(x = z), bins = round(sqrt(length(z_scores))),
                   fill = "red", color = "black", alpha = 0.2) +
    geom_histogram(aes(x = z_log), bins = round(sqrt(length(z_scores))),
                   fill = "blue", color = "black", alpha = 0.2) +
    labs(title = paste("Histogram of Z-scores for", y_name), x = "Z-score", y = "Frequency")
  print(zhist)
}


# Biomarkers
variables <- c("AV45", "ABETA", "PTAU", "TAU")

for (y_name in variables) {
  plot(plot_histogram(all_data, y_name))
  
  box <- boxplot(adni_data[,y_name], main = paste("Boxplot of", y_name), ylab = y_name)
  out_inds <- which(adni_data[,y_name] %in% box$out)
  print(paste("Outliers for", y_name, "are:"))
  print(adni_data[out_inds, c("RID", "VISCODE", y_name)])
}

# Demographics
numeric_vars <- c("AGE", "AV45", "ABETA", "PTAU", "TAU", mrivars, "ADAS13", "MMSE", "PTEDUCAT")

sapply(train_ds[numeric_vars], function(x) c("Mean" = mean(x, na.rm = TRUE), 
                                             "SD" = sd(x, na.rm = TRUE), 
                                             "NAs" = sum(is.na(x)))) -> tr_stats
tr_stats <- as.data.frame(t(round(tr_stats, 2))) 
tr_stats$Miss.ratio <- tr_stats$NAs/nrow(train_ds)

cats <- c("PTGENDER", "APOE4", "DX")

for (var in cats) {
  percs <- table(train_ds[,var])/nrow(train_ds)
  print(paste("Percentages for", var, "are:"))
  print(percs)
  print(sum(is.na(train_ds[,var]))/nrow(train_ds))
}

sapply(test[numeric_vars], function(x) c("Mean" = mean(x, na.rm = TRUE), 
                                         "SD" = sd(x, na.rm = TRUE), 
                                         "NAs" = sum(is.na(x)))) -> te_stats
te_stats <- as.data.frame(t(round(te_stats, 2)))
te_stats$Miss.ratio <- te_stats$NAs/nrow(test)

for (var in cats) {
  percs <- table(test[,var])/nrow(test)
  print(paste("Percentages for", var, "are:"))
  print(percs)
  print(sum(is.na(test[,var]))/nrow(test))
}


