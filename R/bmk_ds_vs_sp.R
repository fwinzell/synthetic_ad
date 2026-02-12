library(dplyr)
library(Rmisc)
library(ggplot2)
library(ggpubr)
library(stringr)
library(ggforce)
library(paletteer)
library(ggsci)

epsilons <- c(200, NA) #c(0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 25, 50, 100, NA)
samples <- c(rep(100, length(epsilons)-length(which(is.na(epsilons)))), 18)
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/bmk_new_deg3_eps", ifelse(is.na(epsilons), "zero", epsilons), ".csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(epsilons)) {
  df <- read.csv(file_paths[i])
  df$Epsilon <- epsilons[i]
  df$samples <- samples[i]
  res_list[[i]] <- df
}

# Combine all data frames
bmk_datasynth <- bind_rows(res_list)

bmk_synthpop <- read.csv("~/Python/WASP-DDLS/SE-benchmark/bmk_synthpop.csv") %>% mutate(
  Epsilon = 0,
  samples = 100
)

util_cols <- c("mutual_inf_diff_value",  "ks_tvd_stat_value", 
               "frac_ks_sigs_value","avg_F1_diff_value", "avg_F1_diff_hout_value", 
               "nnaa_value")
priv_cols <- c("priv_loss_nndr_value", "priv_loss_nnaa_value","hit_rate_value", 
               "avg_nndr_value", "eps_identif_risk_value", "priv_loss_eps_value",  
               "att_discl_risk_value")
cols <- c(util_cols, priv_cols)


bmk_datasynth$Epsilon <- bmk_datasynth %>% 
  select(Epsilon) %>% 
  mutate_all(~replace(., is.na(.), 0)) %>% 
  mutate(Epsilon = factor(Epsilon, levels = as.character(unique(Epsilon)))) %>% 
  pull(Epsilon)  


#rnk_results$util_score <- rnk_results |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, cls_F1_diff, cls_F1_diff_hout, nnaa)) |> rowMeans()

util_df <- rbind(select(bmk_datasynth, dataset, model, all_of(util_cols)),
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

priv_df <- rbind(select(bmk_datasynth, dataset, model, all_of(priv_cols)),
                 select(bmk_synthpop, dataset, model, all_of(priv_cols))) |>
  dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                priv_loss_eps = 1 - abs(priv_loss_eps_value),
                hit_rate = 1 - hit_rate_value,
                eps_identif_risk = 1 - eps_identif_risk_value,
                #mia_recall = 1 - mia_recall_value,
                att_discl_risk = 1 - att_discl_risk_value) 
priv_df$priv_score <- priv_df |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, att_discl_risk)) |> rowMeans()

up_df <- util_df |> dplyr::select(c(dataset, model, util_score)) |> 
  dplyr::inner_join(priv_df |> dplyr::select(c(dataset, model, priv_score)), by = c("dataset", "model"))

up_df$Method <- factor(up_df$dataset, labels=c("DS w DP", "DS w/o DP", "Synthpop"))

up_plot <- ggplot(data = up_df, aes(x=util_score, y=priv_score, fill=Method)) +
  geom_point(pch=21, color = "black", size=3) +
  labs(title = "Utility vs Privacy score", x = "Utility", y = "Privacy") +
  scale_fill_paletteer_d("rcartocolor::BluYl") +
  ylim(0,1) + xlim(0,1)

plot(up_plot)

#ggsave("~/R/DDLS-plots/up_plot_methods.png", up_plot, width = 5, height = 4, units = "in", dpi = 300)


