library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)
#library(Microsoft365R)

########## Privacy ############
df <- read_excel("~/Python/WASP-DDLS/results_final.xlsx")
df$dname <- df$Dataset

df$Dataset <- c("Synthpop", "CTGAN (default)", "CTGAN (optim)", 
               "TabPFN (1.0)", "TabPFN (0.75)", "TabPFN (0.5)", "TabPFN (0.25)",
               "DS (5)", "DS (10)", "DS (50)", "DS (100)", "DS (200)", "DS w/o DP")

real_df <- read_excel("~/Python/WASP-DDLS/results.xlsx")
real_df <- real_df |> filter(Epsilon == "Real") |> mutate(
  Dataset = "Real",
  dname = "real_data"
) |> select(-Epsilon)


########### Correlations ###########
df <- rbind(df, real_df)
samples <- c(rep(100, nrow(df)-2), 18, 100)

cog_df <- df |> select("Dataset", "ADAS ~ MMSE Mean" , "ADAS ~ MMSE Std") 
colnames(cog_df) <- c("Dataset", "Mean", "StD")
cog_df$Variables <- "ADAS ~ MMSE"
cog_df$Mean <- abs(as.numeric(cog_df$Mean))
cog_df$Samples <- samples

tau_df <- df |> select("Dataset", "PTAU ~ TAU Mean", "PTAU ~ TAU Std")
colnames(tau_df) <- c("Dataset", "Mean", "StD")
tau_df$Variables <- "PTAU ~ TAU"
tau_df$Samples <- samples

corr_df <- rbind(cog_df, tau_df) 
corr_df |> mutate(across(.cols = c(Mean, StD), as.numeric)) -> corr_df
corr_df |> 
  mutate(lower = Mean - 1.96 * StD / sqrt(Samples),
         upper = Mean + 1.96 * StD / sqrt(Samples)) -> corr_df

#corr_df$Dataset <- as.factor(ifelse(
#  is.na(corr_df$Eps),
#  corr_df$Method,
#  paste0(corr_df$Method, " (", corr_df$Eps, ")")
#))

order <- c("Real", "Synthpop", "DS w/o DP", 
           "DS (200)", "DS (100)", "DS (50)", "DS (10)", "DS (5)", 
           "CTGAN (optim)", "CTGAN (default)", 
           "TabPFN (1.0)", "TabPFN (0.75)", "TabPFN (0.5)", "TabPFN (0.25)")

corr_df$Dataset <- factor(corr_df$Dataset, levels = order)
corr_df <- arrange(corr_df, Variables, desc(Mean))

# Old
corr_plot <- ggplot(corr_df, aes(x = Dataset, y = Mean, color = Variables, fill=Variables, group = Variables)) +
  geom_bar(stat='identity') +
  geom_errorbar(aes(ymin = lower, ymax = upper), color = "red") +
  #geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, colour=NA) +
  labs(x = "",
       y = "APCC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1),
        legend.position = "bottom") +
  scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6") +
  scale_fill_paletteer_d("ggthemes::Classic_Purple_Gray_6")

# New
corr_plot <- ggplot(corr_df, aes(x = Dataset, y = Mean, fill = Variables)) +
  geom_hline(yintercept = c(1.0, 0.75, 0.5, 0.25), linetype="dotted", color='gray') +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7
  ) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    position = position_dodge(width = 0.8),
    width = 0.2
  ) +
  labs(
    x = "",
    y = "APCC",
    fill = "Variables"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6") +
  scale_fill_paletteer_d("ggthemes::Classic_Purple_Gray_6") +
  theme(legend.position="bottom")

plot(corr_plot)

# %AB+
#df <- rbind(df, real_df)
#samples <- c(samples, 100)

ab_df_cn <- df |> select("Dataset",  "AB+ (CN) Mean", "AB+ (CN) Std") 
ab_df_mci <- df |> select("Dataset", "AB+ (MCI) Mean", "AB+ (MCI) Std")
ab_df_ad <- df |> select("Dataset",  "AB+ (AD) Mean", "AB+ (AD) Std")     
colnames(ab_df_cn) <- c("Dataset",  "Mean", "StD")
colnames(ab_df_mci) <- c("Dataset",  "Mean", "StD")
colnames(ab_df_ad) <- c("Dataset",  "Mean", "StD")
ab_df_cn$DX <- "CN"
ab_df_mci$DX <- "MCI"
ab_df_ad$DX <- "AD"
ab_df <- rbind(ab_df_cn, ab_df_mci, ab_df_ad)
ab_df |> mutate(across(.cols = c(Eps, Mean, StD), as.numeric)) -> ab_df
ab_df |> 
  mutate(samples = rep(samples,3)) |> 
  mutate(lower = Mean - 1.96 * StD / sqrt(samples),
         upper = Mean + 1.96 * StD / sqrt(samples)) -> ab_df

ab_df$DX <- factor(ab_df$DX, levels = c("CN", "MCI", "AD"))

order <- c("Real", "Synthpop", "DS w/o DP", 
           "DS (200)", "DS (100)", "DS (50)", "DS (10)", "DS (5)", 
           "CTGAN (optim)", "CTGAN (default)", 
           "TabPFN (1.0)", "TabPFN (0.75)", "TabPFN (0.5)", "TabPFN (0.25)")
ab_df$Dataset <- factor(ab_df$Dataset, levels = order)

ab_plot <- ggplot(ab_df, aes(x = Dataset, y = Mean, fill = DX)) +
  geom_hline(yintercept = c(100, 75, 50, 25), linetype="dotted", color='gray') +
  geom_bar(stat = "identity", position = "dodge", width=.75, alpha=0.75, color = "black") +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.4, position=position_dodge(width = 0.75), color = "black") +
  #scale_x_continuous(breaks = 1:nrow(df), labels = privacy) + 
  labs(y = "%AB+", x="") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1),
        legend.position = "bottom") +
  scale_fill_brewer(palette = "YlOrRd") +
  scale_color_brewer(palette = "YlOrRd") 
plot(ab_plot)


# APOE4
plot_apoe4 <- function(dx, order) {
  apoe_df <- df %>% select("Dataset", 
                           matches(paste0("APOE4 = [012] \\(", dx, "\\) Mean")), 
                           matches(paste0("APOE4 = [012] \\(", dx, "\\) Std")))
  colnames(apoe_df) <- c("Dataset",  "Mean.0", "Mean.1", "Mean.2", "StD.0", "StD.1", "StD.2")  
  apoe_df$Mean.NAs <- 100 - apoe_df$Mean.0 - apoe_df$Mean.1 - apoe_df$Mean.2
  apoe_df$Dataset <- factor(apoe_df$Dataset, levels = order)
  
  apoe_df |> select("Dataset",  "Mean.0", "Mean.1", "Mean.2", "Mean.NAs") |> 
    pivot_longer(cols = c("Mean.0", "Mean.1", "Mean.2", "Mean.NAs"), names_to = "APOE4", names_prefix = "Mean.", values_to = "Mean") -> apoe_long
  
  apoe_plot <- ggplot(apoe_long, aes(x = Dataset, y = Mean, fill = APOE4)) +
    geom_bar(stat = "identity", position = "stack", width=.75, alpha=0.75, color = "black") +
    #scale_x_continuous(breaks = 1:nrow(df), labels = privacy) + 
    labs(title = paste("APOE4", dx), y = "% Distribution" ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1)) +
    scale_fill_brewer(palette = "YlOrRd") +
    scale_color_brewer(palette = "YlOrRd") 
  return(apoe_plot)
}

apoe4_cn <- plot_apoe4("CN", order)
apoe4_mci <- plot_apoe4("MCI", order)
apoe4_ad <- plot_apoe4("AD", order)

apoe4_plot <- ggarrange(apoe4_cn, apoe4_mci, apoe4_ad, ncol = 3, common.legend = TRUE, legend = "right")
plot(apoe4_plot)

ab_corr <- ggarrange(corr_plot, ab_plot, nrow = 1)
plot(ab_corr)

#ggsave(corr_plot, filename = "/home/fi5666wi/R/DDLS-plots/corr_plot.png", dpi = 500, width = 7, height = 4)
#ggsave(ab_plot, filename = "/home/fi5666wi/R/DDLS-plots/ab_plot.png", dpi = 500, width = 7, height = 4)
ggsave(ab_corr, filename = "/home/fi5666wi/R/DDLS-plots/ab_corr_plot.png", dpi = 600, width = 14, height = 6)
ggsave(apoe4_plot, filename = "/home/fi5666wi/R/DDLS-plots/apoe4_plot.png", dpi = 500, width = 10, height = 6)


#### A4 #####

a4_df <- read_excel("~/Python/WASP-DDLS/results_a4_final.xlsx")
a4_df$dname <- a4_df$Dataset

a4_df$Dataset <- c("CTGAN (default)", "CTGAN (optim)", 
                "TabPFN (1.0)",
                "DS (5)", "DS (10)", "DS (50)", "DS (100)", "DS (200)", "DS w/o DP",
                "Synthpop",
                "Real")

########### Correlations ###########
samples <- c(rep(100, nrow(a4_df)-3), 23, 100, 1)

cogs <- c("MMSCORE ~ DIGITTOTAL", "MMSCORE ~ LDELTOTAL", "LIMMTOTAL ~ LDELTOTAL")
corr_df <- data.frame()
for (i in 1:3) {
  cog_df_i <- a4_df |> select("Dataset", paste0(cogs[i], " Mean") , paste0(cogs[i], " Std")) 
  colnames(cog_df_i) <- c("Dataset", "Mean", "StD")
  cog_df_i$Variables <- cogs[i]
  cog_df_i$Mean <- abs(as.numeric(cog_df_i$Mean))
  cog_df_i$Samples <- samples
  corr_df <- rbind(corr_df, cog_df_i)
}

corr_df |> mutate(across(.cols = c(Mean, StD), as.numeric)) -> corr_df
corr_df |> 
  mutate(lower = Mean - 1.96 * StD / sqrt(Samples),
         upper = Mean + 1.96 * StD / sqrt(Samples)) -> corr_df

#corr_df$Dataset <- as.factor(ifelse(
#  is.na(corr_df$Eps),
#  corr_df$Method,
#  paste0(corr_df$Method, " (", corr_df$Eps, ")")
#))

order <- c("Real", "Synthpop", "DS w/o DP",
           "DS (200)", "DS (100)", "DS (50)", "DS (10)", "DS (5)", 
           "CTGAN (optim)", "CTGAN (default)", "TabPFN (1.0)")

corr_df$Dataset <- factor(corr_df$Dataset, levels = order)
corr_df <- arrange(corr_df, Variables, desc(Mean))

# Old
corr_plot <- ggplot(corr_df, aes(x = Dataset, y = Mean, color = Variables, fill=Variables, group = Variables)) +
  geom_bar(stat='identity') +
  geom_errorbar(aes(ymin = lower, ymax = upper), color = "red") +
  #geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, colour=NA) +
  labs(x = "",
       y = "APCC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1),
        legend.position = "bottom") +
  scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6") +
  scale_fill_paletteer_d("ggthemes::Classic_Purple_Gray_6")

# New
corr_plot <- ggplot(corr_df, aes(x = Dataset, y = Mean, fill = Variables)) +
  geom_hline(yintercept = c(1.0, 0.75, 0.5, 0.25), linetype="dotted", color='gray') +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7
  ) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    position = position_dodge(width = 0.8),
    width = 0.2
  ) +
  labs(
    x = "",
    y = "APCC",
    fill = "Variables"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6") +
  scale_fill_paletteer_d("ggthemes::Classic_Purple_Gray_6") +
  theme(legend.position="bottom")

plot(corr_plot)

# %AB+
#df <- rbind(df, real_df)
#samples <- c(samples, 100)

ab_df <- a4_df |> select("Dataset",  "AB+ Mean", "AB+ Std") 
colnames(ab_df) <- c("Dataset",  "Mean", "StD")

ab_df |> 
  mutate(samples = samples) |> 
  mutate(lower = Mean - 1.96 * StD / sqrt(samples),
         upper = Mean + 1.96 * StD / sqrt(samples)) -> ab_df

ab_df$Dataset <- factor(ab_df$Dataset, levels = order)

ab_plot <- ggplot(ab_df, aes(x = Dataset, y = Mean, fill="darkred")) +
  geom_hline(yintercept = c(100, 75, 50, 25), linetype="dotted", color='gray') +
  geom_bar(stat = "identity", position = "dodge", width=.75, alpha=0.75, color = "black") +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.4, position=position_dodge(width = 0.75), color = "black") +
  #scale_x_continuous(breaks = 1:nrow(df), labels = privacy) + 
  labs(y = "%AB+", x="") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, vjust=1, hjust=1), 
        legend.position = "none")
plot(ab_plot)

ab_corr_a4 <- ggarrange(corr_plot, ab_plot, nrow = 1)
plot(ab_corr_a4)

ggsave(ab_corr_a4, filename = "/home/fi5666wi/R/DDLS-plots/ab_corr_a4_plot.png", dpi = 600, width = 14, height = 6)



