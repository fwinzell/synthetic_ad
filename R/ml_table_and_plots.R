library(ggplot2)
library(dplyr)
library(paletteer)
library(tidyr)
library(ggpubr)

# This script can be used to generate a table over ML results 
# and make some plots to visualize the results

load_results <- function(dataset) {
  # DataSynthesizer
  epsilons <- c(5, 10, 50, 100, 200, NA) #c(0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 25, 50, 100, NA)
  base_dir <- paste0("~/Python/WASP-DDLS/ML-results/", dataset)
  file_paths <- paste0(base_dir, "/deg2_eps_", ifelse(is.na(epsilons), "zero", epsilons), ".csv")
  
  # Read CSVs in a loop
  res_list <- list()
  for (i in seq_along(epsilons)) {
    df <- read.csv(file_paths[i])
    df$Eps <- epsilons[i]
    df$method <- "DS"
    df$Name <- ifelse(is.na(epsilons[i]), "DS w/o DP", paste0("DS (", epsilons[i], ")")) 
    res_list[[i]] <- df
  }
  
  ds_df <- bind_rows(res_list)
  
  # Real data
  real_df <-cbind(data.frame(method = "Real", Name="Real", Eps = NA), read.csv(paste0(base_dir, "/real_data.csv")))
  
  # CTGAN
  epochs <- c(750)
  settings <- c('default', 'optim')
  file_paths <- paste0(base_dir, "/ctgan_", settings, "_epochs_", epochs, ".csv")
  
  # Read CSVs in a loop
  res_list <- list()
  for (i in seq_along(epochs)) {
    df <- read.csv(file_paths[i])
    df$Eps <- epochs[i]
    df$method <- "CTGAN"
    df$Name <- "CTGAN"
    #df$Eps <- NA
    res_list[[i]] <- df
  }
  
  ctgan_df <- bind_rows(res_list)
  
  # Synthpop
  synthpop_df <- read.csv(paste0(base_dir, "/synthpop.csv"))
  synthpop_df$method <- "Synthpop"
  synthpop_df$Eps <- NA
  synthpop_df$Name <- "Synthpop"
  
  # TabPFN
  tabpfn_df <- read.csv(paste0(base_dir, "/tabpfn_t_1.0.csv"))
  tabpfn_df$method <- "TabPFN"
  tabpfn_df$Eps <- NA
  tabpfn_df$Name <- "TabPFN"
  
  #TableDiffusion
  td_epsilons = c('1.0', '10.0', '25.0', '50.0', '100.0')
  file_paths <- paste0(base_dir, "/td_epsilon_", td_epsilons, ".csv")
  
  # Read CSVs in a loop
  res_list <- list()
  for (i in seq_along(td_epsilons)) {
    df <- read.csv(file_paths[i])
    df$Eps <- as.numeric(td_epsilons[i])
    df$method <- "TD"
    df$Name <- paste0("TD (", as.numeric(td_epsilons[i]), ")")
    res_list[[i]] <- df
  }
  td_df <- bind_rows(res_list)
  
  # Combine all data frames
  combined <- rbind(
    real_df,
    select(ds_df, all_of(colnames(real_df))),
    select(td_df, all_of(colnames(td_df))),
    select(ctgan_df, all_of(colnames(real_df))),
    select(synthpop_df, all_of(colnames(real_df))),
    select(tabpfn_df, all_of(colnames(real_df)))
  )
  
  return(combined)
}

#### ADNI ####
adni_df <- load_results("adni")
a4_df <- load_results("a4")

adni_df$n <- 100
adni_df[which(adni_df$method == "DS" & is.na(adni_df$Eps)), "n"] <- 18
adni_df$Dataset <- "ADNI"

a4_df$n <- 100
a4_df[which(a4_df$method == "DS" & is.na(a4_df$Eps)), "n"] <- 23
a4_df$Dataset <- "A4"
 
combined <- rbind(adni_df, a4_df) 
combined  %>% mutate(ci_lower = mean - qt(0.975, df = n - 1) * std / sqrt(n),
                    ci_upper = mean + qt(0.975, df = n - 1) * std / sqrt(n)) -> combined



combined %>% filter(metric %in% c('auc', 'rmse')) %>% arrange(desc(model)) %>% 
  select(-c(metric, std, Dataset, n)) %>% 
  pivot_wider(values_from=c(mean, ci_lower, ci_upper), 
              names_from = label, 
              names_sep = "_", 
              names_sort = T, 
              names_glue = "{label}_{.value}") %>% mutate(Eps = as.integer(Eps)) -> xt # %>% 
  #select(method, model, Eps, DX_mean, DX_std, AB_mean, AB_std, ADAS13_mean, ADAS13_std, MMSE_mean, MMSE_std) -> xt



format_pm <- function(mean, lwr, upr) {
  sprintf("%.3f (%.3f-%.3f)", mean, lwr, upr)
}

mean_cols <- grep("_mean$", names(xt), value = TRUE)
upr_cols   <- gsub("_mean$", "_ci_upper", mean_cols)
lwr_cols   <- gsub("_mean$", "_ci_lower", mean_cols)

table_df <- data.frame(lapply(seq_along(mean_cols), function(i) {
  format_pm(xt[[mean_cols[i]]], xt[[lwr_cols[i]]], xt[[upr_cols[i]]])
}))
names(table_df) <- gsub("_mean$", "", mean_cols)
table_df <- xt %>% select(method, Eps) %>% cbind(table_df)
table_df <- select(table_df, method, Eps, DX, AB, ADAS13, MMSE, AB_status, MMSCORE)

#library(openxlsx)
#write.xlsx(table_df, file = "~/Python/WASP-DDLS/ml_table.xlsx", rowNames = FALSE)

#knitr::kable(latex_df, format = "latex", escape = FALSE, booktabs = TRUE)

welch_t_test <- function(real, syn) {
  se  <- sqrt(real$std^2/real$n + syn$std^2/syn$n)
  t   <- (real$mean - syn$mean) / se
  
  df <- (real$std^2/real$n + syn$std^2/syn$n)^2 /
    ((real$std^2/real$n)^2/(real$n-1) +
       (syn$std^2/syn$n)^2/(syn$n-1))
  
  p_value <- 2 * pt(-abs(t), df)
  return(p_value)
}

# Significance test
significant <- data.frame()
for(m in c("svm", "hgb")) {
  for(l in c("DX", "AB", "AB_status")) {
    combined %>% filter(model == m, label == l, metric == "auc") -> df
    real <- subset(df, method == "Real")
    syn <- subset(df, method != "Real")
    syn$ci.bound <- (real$mean - syn$mean) - 1.96 * sqrt((real$std^2/real$n) + (syn$std^2/syn$n))
    syn %>% filter(ci.bound < 0) %>% rbind(significant) -> significant
  }
}

for(m in c("svm", "hgb")) {
  for(l in c("ADAS13", "MMSE", "MMSCORE")) {
    combined %>% filter(model == m, label == l, metric == "rmse") -> df
    real <- subset(df, method == "Real")
    syn <- subset(df, method != "Real")
    syn$ci.bound <- (real$mean - syn$mean) + 1.0 * sqrt((real$std^2/real$n) + (syn$std^2/syn$n))
    syn %>% filter(ci.bound > 0) %>% rbind(significant) -> significant
  }
}

# Plot for presentation

plot_df <- combined %>% mutate(
  label = recode(label, "AB_status" = "AB", "MMSCORE" = "MMSE")) %>% 
  filter(metric %in% c("auc", "rmse"))

real_scores <- plot_df %>% filter(method=="Real") %>%
  select(model, label, metric, mean, Dataset) %>%
  dplyr::rename(real_mean = mean)

label_list = c("Real", 
               "DS (5)", "DS (10)","DS (50)","DS (100)", 
               "DS (200)", "DS w/o DP", "CTGAN",
               "Synthpop", 
               "TabPFN",
               "TD (1)", "TD (10)", "TD (25)", "TD (50)", "TD (100)")

plot_df$Name <- factor(plot_df$Name, levels = label_list)

auc_df <- plot_df %>% filter(metric=="auc") %>% 
  left_join(real_scores, by=c("model", "label", "metric", "Dataset")) %>%
  mutate(rel_mean = mean/real_mean,
         diff_mean = mean - real_mean) %>%
  mutate(Task = paste0(toupper(model), "-", label, "\n", Dataset))

ggplot(
  auc_df,
  aes(Task, Name, fill = rel_mean)
) +
  geom_tile() +
  geom_text(aes(label = round(mean, 3))) +
  scale_fill_gradient(
    low = "white",
    high = "blue2",
    limits = c(0.75, 1.0),
    oob = scales::squish
  ) + 
  scale_x_discrete(position = "top") +
  theme_minimal() +
  theme(legend.position = "none")

rmse_df <- plot_df %>% filter(metric=="rmse") %>% 
  left_join(real_scores, by=c("model", "label", "metric", "Dataset")) %>%
  mutate(rel_mean = real_mean/mean,
         diff_mean = mean - real_mean) %>%
  mutate(Task = paste0(toupper(model), "-", label, "\n", Dataset))

ggplot(
  rmse_df,
  aes(Task, Name, fill = rel_mean)
) +
  geom_tile() +
  geom_text(aes(label = round(mean, 3))) +
  scale_fill_gradient(
    low = "white",
    high = "blue2",
    limits = c(0.75, 1.0),
    oob = scales::squish
  ) + 
  scale_x_discrete(position = "top") +
  theme_minimal() +
  theme(legend.position = "none")

make_ml_plots <- function(ml_df, xvar = "Name") {
  dx_plot <- ml_df %>% filter(label == 'DX' & metric == 'auc') %>%
    ggplot(aes(x=.data[[xvar]], y=mean, color=model, group=model)) + 
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.1) +
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
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.1) +
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
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.1) +
    labs(title = expression(paste("ADAS13 ", R^2, " score")), x = xvar, y = "R2") +
    scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6", 
                            name = "Model", labels = c("HGBoost", "SVM"))  +
    theme_minimal()
  #plot(adas_plot)
  
  mmse_plot <- ml_df %>% filter(label == 'MMSE' & metric == 'r2') %>%
    ggplot(aes(x=.data[[xvar]], y=mean, color=model, group=model)) + 
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.1) +
    labs(title = expression(paste("MMSE ", R^2, " score")), x = xvar, y = "R2") +
    scale_color_paletteer_d("ggthemes::Classic_Purple_Gray_6", 
                            name = "Model", labels = c("HGBoost", "SVM")) +
    theme_minimal()
  #plot(mmse_plot)
  
  return(list(dx_plot, ab_plot, adas_plot, mmse_plot))
}

plots <- make_ml_plots(combined)

ml_plot <- ggarrange(plotlist = plots, ncol = 2, nrow = 2, common.legend = TRUE, legend = "right")
plot(ml_plot)




