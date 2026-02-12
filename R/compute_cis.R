library(dplyr)
library(tidyr)
library(Rmisc)
library(ggplot2)
library(ggpubr)
library(stringr)
library(ggforce)
library(paletteer)
library(ggsci)

epsilons <- c(5, 10, 25, 50, 100, 200, NA) 
samples <- c(rep(100, length(epsilons)-length(which(is.na(epsilons)))), 18)
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/bmk_new_deg2_eps", ifelse(is.na(epsilons), "zero", epsilons), ".csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(epsilons)) {
  df <- read.csv(file_paths[i])
  df$Epsilon <- epsilons[i]
  df$samples <- samples[i]
  res_list[[i]] <- df
}

# Combine all data frames
bmk_ds <- bind_rows(res_list)


util_cols <- c("mutual_inf_diff_value",  "ks_tvd_stat_value", 
               "frac_ks_sigs_value","avg_F1_diff_value", "avg_F1_diff_hout_value", 
               "nnaa_value")
priv_cols <- c("priv_loss_nndr_value", "priv_loss_nnaa_value","hit_rate_value", 
               "avg_nndr_value", "eps_identif_risk_value", "priv_loss_eps_value",  
               "mia_recall_value", "att_discl_risk_value")
cols <- c(util_cols, priv_cols)


bmk_ds$Epsilon <- bmk_ds %>% 
  select(Epsilon) %>% 
  mutate_all(~replace(., is.na(.), 0)) %>% 
  mutate(Epsilon = factor(Epsilon, levels = as.character(unique(Epsilon)))) %>% 
  pull(Epsilon)  


compute_stats <- function(dataset_name, var_name) {
  df <- bmk_ds %>% filter(dataset == dataset_name)
  
  x <- df[[var_name]]  # safely extract the column by name
  n <- sum(!is.na(x))
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  error_margin <- qt(0.975, df = n - 1) * s / sqrt(n)
  ci_lower <- m - error_margin
  ci_upper <- m + error_margin
  
  return(tibble(
    dataset = dataset_name,
    variable = var_name,
    n = n,
    mean = m,
    sd = s,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  ))
}


res <- rbind(compute_stats("degree2_eps_zero", "eps_identif_risk_value"),
             compute_stats("degree2_eps_zero", "mia_recall_value")
)

print(res)

res <- data.frame()
for (dset in unique(bmk_ds$dataset)){
  for (col in priv_cols) {
    res <- rbind(res, compute_stats(dset, col))
  }
}

res %>% filter(variable == "att_discl_risk_value") %>% print()


#### CTGAN ####
epochs <- c(10, 50, 100) 
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/bmk_ctgan_epochs_", epochs, ".csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(epochs)) {
  df <- read.csv(file_paths[i])
  df$Epochs <- epochs[i]
  df$samples <- 100
  res_list[[i]] <- df
}

# Combine all data frames
bmk_ctgan <- bind_rows(res_list)

bmk_ctgan$Epochs <- as.factor(bmk_ctgan$Epochs)


#### Synthpop ####
bmk_synthpop <- read.csv("~/Python/WASP-DDLS/SE-benchmark/bmk_synthpop_2.csv") %>% mutate(
  Epochs = 0,
  samples = 100
)

util_df <- rbind(select(bmk_ds, dataset, model, all_of(util_cols)),
                 select(bmk_ctgan, dataset, model, all_of(util_cols)),
                 select(bmk_synthpop, dataset, model, all_of(util_cols))) |>
  dplyr::mutate(avg_F1_diff_value = abs(avg_F1_diff_value),
                avg_F1_diff_hout_value = abs(avg_F1_diff_hout_value)) |>
  dplyr::mutate(mutual_inf_diff = 1 - tanh(mutual_inf_diff_value),
                ks_tvd_stat = 1 - ks_tvd_stat_value,
                frac_ks_sigs = 1 - frac_ks_sigs_value,
                avg_F1_diff = 1 - abs(avg_F1_diff_value),
                avg_F1_diff_hout = 1 - abs(avg_F1_diff_hout_value),
                nnaa = 1 - nnaa_value) 
util_df$util_score <- util_df |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, avg_F1_diff, avg_F1_diff_hout, nnaa)) |> rowMeans()

util_df$Method <- factor(util_df$dataset, levels = unique(util_df$dataset),
                         labels=c("DS (5)", "DS (10)", "DS (25)","DS (50)","DS (100)", 
                                  "DS (200)", "DS w/o DP", "CTGAN (10)", 
                                  "CTGAN (50)", "CTGAN (100)", "Synthpop"))

priv_df <- rbind(select(bmk_ds, dataset, model, all_of(priv_cols)),
                 select(bmk_ctgan, dataset, model, all_of(priv_cols)),
                 select(bmk_synthpop, dataset, model, all_of(priv_cols))) |>
  dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                priv_loss_eps = 1 - abs(priv_loss_eps_value),
                hit_rate = 1 - hit_rate_value,
                eps_identif_risk = 1 - eps_identif_risk_value,
                mia_recall = 1 - mia_recall_value,
                att_discl_risk = 1 - att_discl_risk_value) 
priv_df$priv_score <- priv_df |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, att_discl_risk)) |> rowMeans()

priv_df$Method <- factor(priv_df$dataset, levels = unique(priv_df$dataset),
                         labels=c("DS (5)", "DS (10)", "DS (25)","DS (50)","DS (100)", 
                                  "DS (200)", "DS w/o DP", "CTGAN (10)", 
                                  "CTGAN (50)", "CTGAN (100)", "Synthpop"))

up_df <- util_df |> dplyr::select(c(Method, model, util_score)) |> 
  dplyr::inner_join(priv_df |> dplyr::select(c(Method, model, priv_score)), by = c("Method", "model"))



up_df %>% dplyr::group_by(Method) %>% dplyr::summarise(Mean_util = mean(util_score), 
                                                       SD_util  = sd(util_score),
                                                       Mean_priv = mean(priv_score),
                                                       SD_priv = sd(priv_score),
                                                       N = n()) %>% mutate(uci_lower = Mean_util - qt(0.975, df = N - 1) * SD_util / sqrt(N),
                                                                           uci_upper = Mean_util + qt(0.975, df = N - 1) * SD_util / sqrt(N),
                                                                           pci_lower = Mean_priv - qt(0.975, df = N - 1) * SD_priv / sqrt(N),
                                                                           pci_upper = Mean_priv + qt(0.975, df = N - 1) * SD_priv / sqrt(N),) -> bop
print(bop)

compute_stats <- function(df, var_name) {
  res <- df %>% dplyr::group_by(Method) %>% dplyr::summarise(Mean = mean(!!sym(var_name)), 
                                                      SD  = sd(!!sym(var_name)),
                                                      N = n()) %>% mutate(ci_lower = Mean - qt(0.975, df = N - 1) * SD / sqrt(N),
                                                                          ci_upper = Mean + qt(0.975, df = N - 1) * SD / sqrt(N))
  return(res)
}

res <- compute_stats(priv_df, "eps_identif_risk_value")


print(compute_stats(priv_df, "mia_recall_value"))
print(compute_stats(priv_df, "att_discl_risk_value"))

res <- compute_stats(priv_df, "mia_recall_value")
res[7,]-res[6,]


util_cols <- c("mutual_inf_diff_value",  "ks_tvd_stat_value", 
               "frac_ks_sigs_value","avg_F1_diff_value", "avg_F1_diff_hout_value", 
               "nnaa_value")

print(compute_stats(util_df, "mutual_inf_diff_value"))
print(compute_stats(util_df, "frac_ks_sigs_value"))
print(compute_stats(util_df, "avg_F1_diff_hout_value"))
