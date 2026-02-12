library(dplyr)
library(tidyr)

folder = "/home/fi5666wi/R/data/DDLS/A4/Assessments"
files = list.files(path = folder, pattern = "\\.csv$", full.names = TRUE)

for(file in files) {
  df <- read.csv(file)
}

ptdemog <- read.csv(paste(folder, "A4_PTDEMOG_PRV2_17Dec2025.csv", sep="/"))

c3comp <- read.csv(paste(folder, "A4_C3COMP_PRV2_17Dec2025.csv", sep="/")) %>%
  filter(VISCODE == 1) 
cdr <- read.csv(paste(folder, "A4_CDR_PRV2_17Dec2025.csv", sep="/")) 
cog_digit <- read.csv(paste(folder, "A4_COGDIGIT_PRV2_17Dec2025.csv", sep="/"))
cog_remind <- read.csv(paste(folder, "A4_COGFCSR16_PRV2_17Dec2025.csv", sep="/"))
cog_logic <- read.csv(paste(folder, "A4_COGLOGIC_PRV2_17Dec2025.csv", sep="/"))
cpath <- read.csv(paste(folder, "A4_CPATH_PRV2_17Dec2025.csv", sep="/")) %>% 
  filter(Question_Text == "Total C-Path Score (total)" & VISCODE == 1)
mmse <- read.csv(paste(folder, "A4_MMSE_PRV2_17Dec2025.csv", sep="/"))
phyneuro <- read.csv(paste(folder, "A4_PHYNEURO_PRV2_17Dec2025.csv", sep="/"))


a4_df <- cdr %>% select(BID, VISCODE, CDSOB) %>% unique() %>% right_join(ptdemog, by = c("BID", "VISCODE"))
a4_df <- mmse %>% select(BID, VISCODE, MMSCORE) %>% unique() %>% right_join(a4_df, by = c("BID", "VISCODE"))
a4_df <- cog_digit %>% select(BID, VISCODE, DIGITTOTAL) %>% unique() %>% right_join(a4_df, by = c("BID", "VISCODE"))
a4_df <- cog_logic %>% select(BID, VISCODE, LIMMTOTAL, LDELTOTAL) %>% unique() %>% right_join(a4_df, by = c("BID", "VISCODE"))
a4_df <- cpath %>% select(BID, VISCODE, score) %>% unique() %>% dplyr::rename(CPATH = score) %>% right_join(a4_df, by = c("BID", "VISCODE"))

a4_df %>% dplyr::summarise(
  dplyr::across(
    where(is.numeric),
    list(min = ~min(.x, na.rm = TRUE),
         max = ~max(.x, na.rm = TRUE))
  )
)

### Imaging

folder = "/home/fi5666wi/R/data/DDLS/A4/Imaging"

files = list.files(path = folder, pattern = "\\.csv$", full.names = TRUE)

amy_pet <- read.csv(paste(folder, "A4_PETSUVR_PRV2_17Dec2025.csv", sep="/")) 
ab_data <- read.csv(paste(folder, "A4_PETVADATA_PRV2_17Dec2025.csv", sep="/"))
tau_pet <- read.csv(paste(folder, "TAUSUVR_PETSURFER_17Dec2025.csv", sep="/"))
vmri <- read.csv(paste(folder, "A4_VMRI_PRV2_17Dec2025.csv", sep="/"))


img_df <- ab_data %>% select(BID, PMODSUVR, SCORE) %>% unique() %>% dplyr::rename(AB_status = SCORE)
img_df <- vmri %>% select(BID, LeftCorticalGrayMatter, RightCorticalGrayMatter, 
                          LeftHippocampus, RightHippocampus, LeftEntorhinal, RightEntorhinal, 
                          LeftFusiform, RightFusiform) %>% unique() %>%
  full_join(img_df, by = "BID")

## Remove Imaging outliers
img_data <- img_df %>% select(-c(BID, PMODSUVR, AB_status))
log_img_data <- mutate_all(img_data, log)

z_threshold <- 4
for (y_name in colnames(img_data)) {
  z_scores <- scale(img_data[,y_name])
  n_outliers <- sum(z_scores > z_threshold | z_scores < -z_threshold, na.rm = TRUE)
  
  z_log <- scale(log_img_data[,y_name])
  n_log_outliers <- sum(z_log > z_threshold | z_log < -z_threshold, na.rm = TRUE)
  
  if (n_outliers > n_log_outliers) {
    z_scores <- z_log
    n_outliers <- n_log_outliers
  }
  print(paste("Outliers for", y_name, ": ", n_outliers))
  
  img_data[which(z_scores > z_threshold | z_scores < -z_threshold), y_name] <- NA
  log_img_data[which(z_scores > z_threshold | z_scores < -z_threshold), y_name] <- NA
}

#adni_data[, mrivars] <- mri_data

img_df[, colnames(img_data)] <- img_data
 

a4_df <- img_df %>% right_join(a4_df, by = "BID")

a4_df <- a4_df %>% drop_na(AB_status) %>% select(-c(PTRACE, PTLANG, PTPLANG, update_stamp))

duplicated_rows <- a4_df$BID %in% a4_df$BID[duplicated(a4_df$BID)]

a4_df <- a4_df[!duplicated(a4_df$BID), ]

# split into test and train sets
smp_size <- floor(0.80 * nrow(a4_df))

## set the seed to make your partition reproducible
set.seed(1)
train_ind <- sample(seq_len(nrow(a4_df)), size = smp_size)
a4_df <- select(a4_df, -VISCODE)

X_train <- a4_df[train_ind, ]
X_test <- a4_df[-train_ind, ]

write.csv(a4_df, file = "~/R/data/DDLS/a4_all.csv", row.names = FALSE)
write.csv(X_train, file = "~/R/data/DDLS/a4_train.csv", row.names = FALSE)
write.csv(X_test, file = "~/R/data/DDLS/a4_test.csv", row.names = FALSE)


X_train %>% select(-c(EXAMDAY)) %>% write.csv(file = "~/R/data/DDLS/a4_train_ml.csv", row.names = FALSE)



