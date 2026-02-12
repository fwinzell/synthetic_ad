library(ggplot2)
library(dplyr)
library(paletteer)
library(tidyr)
library(ggpubr)

epsilons <- c(5, 10, 25, 50, 100, 200, NA) #c(0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 25, 50, 100, NA)
samples <- c(rep(100, length(epsilons)-1), 18)
file_paths <- paste0("~/Python/WASP-DDLS/ML-results/deg2_eps_", ifelse(is.na(epsilons), "zero", epsilons), ".csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(epsilons)) {
  df <- read.csv(file_paths[i])
  df$Epsilon <- epsilons[i]
  df$samples <- samples[i]
  res_list[[i]] <- df
}

# Combine all data frames
res_df <- bind_rows(res_list)

res_df$Privacy <- factor(
  ifelse(is.na(res_df$Epsilon), "NA", res_df$Epsilon), 
  levels = c("NA", sort(unique(res_df$Epsilon[!is.na(res_df$Epsilon)]), decreasing = TRUE))
)
eps <- levels(res_df$Privacy)

#eps <- df$Epsilon
#eps <- eps[!is.na(eps)]
#privacy <- c("Real", paste("eps:", eps))
res_df <- rbind(res_df, cbind(data.frame(Epsilon = NA, samples = 100, Privacy = "Real"), read.csv("~/Python/WASP-DDLS/ML-results/real_data.csv"))) 
res_df$Privacy <- factor(
  res_df$Privacy,
  levels = c("Real", eps)
)
str(res_df)

res_df |> 
  mutate(lower = mean - 1.96 * std / sqrt(samples),
         upper = mean + 1.96 * std / sqrt(samples)) -> ml_df

make_ml_plots <- function(ml_df, xvar = "Privacy") {
  dx_plot <- ml_df %>% filter(label == 'DX' & metric == 'auc') %>%
    ggplot(aes(x=.data[[xvar]], y=mean, color=model, group=model)) + 
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.1) +
    labs(title = "Diagnosis AUC (1 vs rest)", x = xvar, y = "AUC") + 
    scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6",
                            name = "Model", labels = c("HGBoost", "SVM")) +
    ylim(0, 1) +
    theme_minimal()
  #plot(dx_plot)
  
  ab_plot <- ml_df %>% filter(label == 'AB' & metric == 'auc') %>%
    ggplot(aes(x=.data[[xvar]], y=mean, color=model, group=model)) + 
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.1) +
    labs(title = "AB+/- AUC", x = xvar, y = "AUC") + 
    scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6",  
                            name = "Model", labels = c("HGBoost", "SVM")) +
    ylim(0, 1) +
    theme_minimal()
  #plot(ab_plot)
  
  adas_plot <- ml_df %>% filter(label == 'ADAS13' & metric == 'r2') %>%
    ggplot(aes(x=.data[[xvar]], y=mean, color=model, group=model)) + 
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.1) +
    labs(title = expression(paste("ADAS13 ", R^2, " score")), x = xvar, y = "R2") +
    scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6", 
                            name = "Model", labels = c("HGBoost", "SVM"))  +
    theme_minimal()
  #plot(adas_plot)
  
  mmse_plot <- ml_df %>% filter(label == 'MMSE' & metric == 'r2') %>%
    ggplot(aes(x=.data[[xvar]], y=mean, color=model, group=model)) + 
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.1) +
    labs(title = expression(paste("MMSE ", R^2, " score")), x = xvar, y = "R2") +
    scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6", 
                            name = "Model", labels = c("HGBoost", "SVM")) +
    theme_minimal()
  #plot(mmse_plot)
  
  return(list(dx_plot, ab_plot, adas_plot, mmse_plot))
}

plots <- make_ml_plots(ml_df)

ml_plot <- ggarrange(plotlist = plots, ncol = 2, nrow = 2, common.legend = TRUE, legend = "right")
plot(ml_plot)


ggsave("~/R/DDLS-plots/ml_plot.png", ml_plot, width = 7, height = 5, units = "in", dpi = 500)


ml_df %>% filter(metric %in% c('auc','rmse')) %>% 
  group_by(Privacy, label, metric) %>%
  slice_max(order_by = mean, n = 1) %>% ungroup() -> best_df


best_df %>% select(Privacy, label, metric, mean, std) %>% pivot_wider(names_from = c(metric, label), values_from = c(mean, std)) -> table_df
table_df <- table_df[,c(1,4,8,2,6,3,7,5,9)]
library(xtable)
xtable(table_df, caption = "Best results for each metric and label")


##### Degree #####
degree <- c(1,2,3,4) 
file_paths <- paste0("~/Python/WASP-DDLS/ML-results/deg_", degree, ".csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(epsilons)) {
  df <- read.csv(file_paths[i])
  df$Degree <- degree[i]
  df$samples <- 18
  res_list[[i]] <- df
}

# Combine all data frames
res_df_deg <- bind_rows(res_list)

res_df_deg <- rbind(res_df_deg, cbind(data.frame(samples = 100, Degree = "Real"), read.csv("~/Python/WASP-DDLS/ML-results/real_data.csv"))) 
res_df_deg$Degree <- factor(
  res_df_deg$Degree,
  levels = c("Real", degree)
)

res_df_deg |> 
  mutate(lower = mean - 1.96 * std / sqrt(samples),
         upper = mean + 1.96 * std / sqrt(samples)) -> ml_df_deg

plots_deg <- make_ml_plots(ml_df_deg, xvar="Degree")

ml_plot_deg <- ggarrange(plotlist = plots_deg, ncol = 2, nrow = 2, common.legend = TRUE, legend = "right")
plot(ml_plot_deg)


# DS Synthpop and CTGAN

epochs <- c(10,50,100) 
file_paths <- paste0("~/Python/WASP-DDLS/ML-results/ctgan_epochs_", epochs, ".csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(epochs)) {
  df <- read.csv(file_paths[i])
  df$Epochs <- epochs[i]
  df$samples <- 100
  res_list[[i]] <- df
}

# Combine all data frames
res_df <- bind_rows(res_list)

res_df <- rbind(res_df, cbind(data.frame(samples = 100, Epochs = "Real"), read.csv("~/Python/WASP-DDLS/ML-results/real_data.csv"))) 
res_df$Epochs <- factor(
  res_df$Epochs,
  levels = c("Real", epochs)
)

res_df |> 
  mutate(lower = mean - 1.96 * std / sqrt(samples),
         upper = mean + 1.96 * std / sqrt(samples)) -> ml_df

plots_ctgan <- make_ml_plots(ml_df, xvar="Epochs")

ml_plot_ctgan <- ggarrange(plotlist = plots_ctgan, ncol = 2, nrow = 2, common.legend = TRUE, legend = "right")
plot(ml_plot_ctgan)








