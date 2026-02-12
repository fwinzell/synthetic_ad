library(synthpop)
library(dplyr)
library(ggplot2)
library(ggpubr)

adni_train <- read.csv("~/R/data/DDLS/adni_plus_train.csv")
a4_train <- read.csv("~/R/data/DDLS/a4_train.csv")

vars <- colnames(adni_train)[-1]
seq <- sample(1:length(vars), replace=FALSE)
adni_train |> select(-RID) |> syn(method="cart", seed=1, visit.sequence = seq) -> syn_test

syn_df <- syn_test$syn


# Correlation coefficients
# Pearson: cov(X, Y) / (sd(X) * sd(Y))
kr <- -cor(adni_train$ADAS13, adni_train$MMSE, use = "complete.obs", method = "pearson")
ks <- -cor(syn_df$ADAS13, syn_df$MMSE, use = "complete.obs", method = "pearson")

# plot ADAS13 vs MMSE
colors <- c("CN" = "blue", "MCI" = "purple", "Dementia" = "red")

gpr <- ggplot(data = subset(adni_train, !is.na(DX)), aes(x = ADAS13, y = MMSE, color = DX)) + 
  geom_point() + geom_smooth(method = "lm", se = FALSE) +
  ggtitle(sprintf("Corr: %f", kr)) +
  scale_y_continuous(trans = "reverse") +
  scale_color_manual(values = colors)

gps <- ggplot(data = subset(syn_df, !is.na(DX)), aes(x = ADAS13, y = MMSE, color = DX)) + 
  geom_point() + geom_smooth(method = "lm", se = FALSE) +
  ggtitle(sprintf("Corr: %f", ks)) +
  scale_y_continuous(trans = "reverse") +
  scale_color_manual(values = colors)

fig <- ggarrange(gpr, gps, ncol = 2)
print(fig)

# Generate data (ADNI)
adni_train <- select(adni_train, -DXN)
vars <- colnames(adni_train)[-1]
categorical_vars <- c("PTGENDER", "APOE4", "DX", "PTEDUCAT")
adni_train[categorical_vars] <- lapply(adni_train[categorical_vars], factor)

for (s in 1:100) {
  seq <- sample(1:length(vars), replace=FALSE)
  adni_train |> select(-RID) |> syn(method="cart", seed=s, visit.sequence = seq, minnumlevels=1) -> gen_data
  
  syn_df <- gen_data$syn
  
  write.csv(syn_df, sprintf("~/Python/WASP-DDLS/synthpop/adni_plus/syn_ranseq_seed_%d.csv", s), row.names = FALSE)
}

# Generate data (A4)
vars <- colnames(a4_train)[-1]
categorical_vars <- c("PTGENDER", "AB_status", "PTETHNIC", "PTMARRY", "PTNOTRT", "PTHOME")
a4_train[categorical_vars] <- lapply(a4_train[categorical_vars], factor)

for (s in 1:100) {
  seq <- sample(1:length(vars), replace=FALSE)
  a4_train |> select(-BID) |> syn(method="cart", seed=s, visit.sequence = seq, minnumlevels=1) -> gen_data
  
  syn_df <- gen_data$syn
  
  write.csv(syn_df, sprintf("~/Python/WASP-DDLS/synthpop/a4/syn_ranseq_seed_%d.csv", s), row.names = FALSE)
}
