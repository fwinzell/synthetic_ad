library(ggplot2)

#synthetic <- read.csv("~/R/data/DDLS/adni_synthetic_bl_2430.csv", header = TRUE, stringsAsFactors = TRUE)
synthetic <- read.csv("~/Python/WASP-DDLS/DS-synthetic-data/degree3_root_AGE/bn_eps_100.csv", header = TRUE, stringsAsFactors = TRUE)
synthetic$DX[which(synthetic$DX == "")] <- NA
real <- read.csv("~/R/data/DDLS/adni_train_ds.csv", header = TRUE, stringsAsFactors = TRUE)

library(ppcor)

dfr <- na.omit(real[, c("ABETA", "TAU", "PTAU", "AV45")])

#partial <- pcor.test(dfr$ABETA, dfr$TAU, dfr$PTAU)
corr <- cor(dfr$PTAU, dfr$TAU, use = "complete.obs", method = "pearson")

dfr$AB <- "AB-" 
dfr$AB[which(dfr$AV45 > 1.11)] <- "AB+"
dfr$AB <- factor(dfr$AB, levels = c("AB-", "AB+"))

gpr <- ggplot(data = dfr, aes(x = PTAU, y = TAU, color = AB)) + 
  geom_point() + geom_smooth(method = "lm", se = FALSE)

print(gpr)

dfs <- na.omit(synthetic[, c("ABETA", "TAU", "PTAU", "AV45")])
cors <- cor(dfs$PTAU, dfs$TAU, use = "complete.obs", method = "pearson")

dfs$AB <- "AB-" 
dfs$AB[which(dfs$AV45 > 1.11)] <- "AB+"
dfs$AB <- factor(dfs$AB, levels = c("AB-", "AB+"))

gps <- ggplot(data = dfs, aes(x = PTAU, y = TAU, color = AB)) + 
  geom_point() + geom_smooth(method = "lm", se = FALSE)

print(gps)

print(sprintf("Real: %f", corr))
print(sprintf("Synthetic: %f", cors))

