library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)
#library(Microsoft365R)

########## Privacy ############
df <- read_excel("~/Python/WASP-DDLS/results.xlsx")

#df$samples <- c(NA, 18, rep(100, nrow(df)-2))

#df_srd <- df |> select("Epsilon", "Mean SRD", "...10", "samples") |> 
#  rename("Average" = "Mean SRD", "StD" = "...10") |> 
#  filter(!row_number() %in% c(1,2)) |> mutate_all(as.numeric)

#df_srd$lower <- df_srd$Average - 1.96 * df_srd$StD / sqrt(df_srd$samples)
#df_srd$upper <- df_srd$Average + 1.96 * df_srd$StD / sqrt(df_srd$samples)

########### Correlations ###########
df <- df |> filter(Epsilon %in% c("Real", "zero", "200", "100", "50", "25", "10", "5")) #read_excel("/Users/filipwinzell/WASP-DDLS/results-ddls-new-mac.xlsx", sheet = "correlations")
samples <- c(100, 18, rep(100, nrow(df)-2))
eps <- df$Epsilon
eps <- c(0, eps[!is.na(as.numeric(eps))])
privacy <- c("Real", paste("eps:", eps))


cog_df <- df |> select("Epsilon", "ADAS ~ MMSE Mean" , "ADAS ~ MMSE Std") 
colnames(cog_df) <- c("Epsilon", "Mean", "StD")
cog_df$Variables <- "ADAS ~ MMSE"
cog_df$Mean <- abs(as.numeric(cog_df$Mean))


tau_df <- df |> select("Epsilon", "PTAU ~ TAU Mean", "PTAU ~ TAU Std")
colnames(tau_df) <- c("Epsilon", "Mean", "StD")
tau_df$Variables <- "PTAU ~ TAU"

corr_df <- rbind(cog_df, tau_df) 
corr_df |> mutate(across(.cols = c(Epsilon, Mean, StD), as.numeric)) -> corr_df
corr_df |> 
  mutate(samples = rep(samples,2)) |> 
  mutate(lower = Mean - 1.96 * StD / sqrt(samples),
         upper = Mean + 1.96 * StD / sqrt(samples)) -> corr_df

corr_df$Privacy <- rep(1:nrow(df), 2)

corr_plot_1 <- ggplot(corr_df, aes(x = Privacy, y = Mean, color = Variables)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.1) +
  scale_x_continuous(breaks = 1:nrow(df), labels = privacy) +
  labs(x = "Epsilon",
       y = "Correlation")
plot(corr_plot_1)

corr_plot <- ggplot(corr_df, aes(x = Privacy, y = Mean, color = Variables, fill=Variables)) +
  geom_line() +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, colour=NA) +
  labs(x = "Privacy",
       y = "PCC") +
  scale_x_continuous(breaks = 1:nrow(df), labels = privacy) + 
  theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1)) +
  scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6") +
  scale_fill_paletteer_d("ggthemes::Classic_Purple_Gray_6") +
  theme_minimal()
plot(corr_plot)

# %AB+
ab_df_cn <- df |> select("Epsilon", "AB+ (CN) Mean", "AB+ (CN) Std") 
ab_df_mci <- df |> select("Epsilon", "AB+ (MCI) Mean", "AB+ (MCI) Std")
ab_df_ad <- df |> select("Epsilon", "AB+ (AD) Mean", "AB+ (AD) Std")     
colnames(ab_df_cn) <- c("Epsilon", "Mean", "StD")
colnames(ab_df_mci) <- c("Epsilon", "Mean", "StD")
colnames(ab_df_ad) <- c("Epsilon", "Mean", "StD")
ab_df_cn$DX <- "CN"
ab_df_mci$DX <- "MCI"
ab_df_ad$DX <- "AD"
ab_df <- rbind(ab_df_cn, ab_df_mci, ab_df_ad)
ab_df |> mutate(across(.cols = c(Epsilon, Mean, StD), as.numeric)) -> ab_df
ab_df |> 
  mutate(samples = rep(samples,3)) |> 
  mutate(lower = Mean - 1.96 * StD / sqrt(samples),
         upper = Mean + 1.96 * StD / sqrt(samples)) -> ab_df

ab_df$Privacy <- rep(1:nrow(df), 3)
ab_df$DX <- factor(ab_df$DX, levels = c("CN", "MCI", "AD"))
ab_plot <- ggplot(ab_df, aes(x = Privacy, y = Mean, fill = DX)) +
  geom_bar(stat = "identity", position = "dodge", width=.75, alpha=0.75, color = "black") +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.4, position=position_dodge(width = 0.75), color = "black") +
  scale_x_continuous(breaks = 1:nrow(df), labels = privacy) + 
  labs(title = "%AB+") +
  theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1)) +
  scale_fill_brewer(palette = "YlOrRd") +
  scale_color_brewer(palette = "YlOrRd") 
plot(ab_plot)


# APOE4
plot_apoe4 <- function(dx) {
  apoe_df <- df %>% select("Epsilon",
                           matches(paste0("APOE4 = [012] \\(", dx, "\\) Mean")), 
                           matches(paste0("APOE4 = [012] \\(", dx, "\\) Std")))
  colnames(apoe_df) <- c("Epsilon", "Mean.0", "Mean.1", "Mean.2", "StD.0", "StD.1", "StD.2")  
  apoe_df$Mean.NAs <- 100 - apoe_df$Mean.0 - apoe_df$Mean.1 - apoe_df$Mean.2
  apoe_df$Privacy <- 1:nrow(df)
  
  apoe_df |> select("Epsilon", "Mean.0", "Mean.1", "Mean.2", "Mean.NAs", "Privacy") |> 
    pivot_longer(cols = c("Mean.0", "Mean.1", "Mean.2", "Mean.NAs"), names_to = "APOE4", names_prefix = "Mean.", values_to = "Mean") -> apoe_long
  
  apoe_plot <- ggplot(apoe_long, aes(x = Privacy, y = Mean, fill = APOE4)) +
    geom_bar(stat = "identity", position = "stack", width=.75, alpha=0.75, color = "black") +
    scale_x_continuous(breaks = 1:nrow(df), labels = privacy) + 
    labs(title = paste("APOE4", dx), y = "% Distribution" ) +
    theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1)) +
    scale_fill_brewer(palette = "YlOrRd") +
    scale_color_brewer(palette = "YlOrRd")
  return(apoe_plot)
}

apoe4_cn <- plot_apoe4("CN")
apoe4_mci <- plot_apoe4("MCI")
apoe4_ad <- plot_apoe4("AD")

apoe4_plot <- ggarrange(apoe4_cn, apoe4_mci, apoe4_ad, ncol = 3, common.legend = TRUE, legend = "right")
plot(apoe4_plot)

############### ML results ###############
df <- read_excel("/Users/filipwinzell/WASP-DDLS/results-ddls-new-mac.xlsx", sheet = "ML")
samples <- c(100, 18, rep(100, nrow(df)-2))

svm_df <- df |> select("Epsilon", 4:7) 
colnames(svm_df) <- c("Epsilon", "Acc.Mean", "Acc.StD", "F1.Mean", "F1.StD")
svm_df |> mutate(across(.cols = c(Epsilon, Acc.Mean, Acc.StD, F1.Mean, F1.StD), as.numeric)) -> svm_df
svm_df$Method <- "SVM"

hgb_df <- df |> select("Epsilon", 8:11) 
colnames(hgb_df) <- c("Epsilon", "Acc.Mean", "Acc.StD", "F1.Mean", "F1.StD")
hgb_df |> mutate(across(.cols = c(Epsilon, Acc.Mean, Acc.StD, F1.Mean, F1.StD), as.numeric)) -> hgb_df
hgb_df$Method <- "HGBoost"

ml_df <- rbind(svm_df, hgb_df)
ml_df |> 
  mutate(samples = rep(samples,2)) |> 
  mutate(Acc.lower = Acc.Mean - 1.96 * Acc.StD / sqrt(samples),
         Acc.upper = Acc.Mean + 1.96 * Acc.StD / sqrt(samples)) -> ml_df
ml_df |> 
  mutate(F1.lower = F1.Mean - 1.96 * F1.StD / sqrt(samples),
         F1.upper = F1.Mean + 1.96 * F1.StD / sqrt(samples)) -> ml_df
ml_df$Privacy <- rep(1:13, 2)
ml_df$Method <- as.factor(ml_df$Method)

acc_plot <- ggplot(ml_df, aes(x = Privacy, y = Acc.Mean, color = Method)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = Acc.lower, ymax = Acc.upper), width = 0.1) +
  scale_x_continuous(breaks = 1:13, labels = privacy) +
  labs(x = "",
       y = "Accuracy") +
  theme(axis.text.x = element_blank()) 

f1_plot <- ggplot(ml_df, aes(x = Privacy, y = F1.Mean, color = Method)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = F1.lower, ymax = F1.upper), width = 0.1) +
  scale_x_continuous(breaks = 1:13, labels = privacy) +
  labs(x = "Privacy",
       y = "F1") +
  theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1))

ml_plot <- ggarrange(acc_plot, f1_plot, ncol = 1, common.legend = TRUE, legend = "right")
plot(ml_plot)


ggsave(srd_plot, filename = "/Users/filipwinzell/WASP-DDLS/plots/srd_plot.png", dpi = 500, width = 6, height = 4)
ggsave(corr_plot, filename = "/Users/filipwinzell/WASP-DDLS/plots/corr_plot.png", dpi = 500, width = 7, height = 5)
ggsave(ab_plot, filename = "/Users/filipwinzell/WASP-DDLS/plots/ab_plot.png", dpi = 500, width = 8, height = 4) 
ggsave(apoe4_plot, filename = "/Users/filipwinzell/WASP-DDLS/plots/apoe4_plot.png", dpi = 500, width = 10, height = 6)
ggsave(ml_plot, filename = "/Users/filipwinzell/WASP-DDLS/plots/ml_plot.png", dpi = 500, width = 7, height = 6)
  
  