library(dplyr)
library(Rmisc)
library(ggplot2)
library(ggpubr)
library(stringr)
library(ggforce)
library(paletteer)
library(ggsci)

epochs <- c(10, 50, 100) #c(0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 25, 50, 100, NA)
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
bmk_results <- bind_rows(res_list)

bmk_synthpop <- read.csv("~/Python/WASP-DDLS/SE-benchmark/bmk_synthpop.csv") %>% mutate(
  Epochs = 0,
  samples = 100
)

bmk_results$Epochs <- as.factor(bmk_results$Epochs)

# Mean - SD - Confidence intervals
util_cols <- c("mutual_inf_diff_value",  "ks_tvd_stat_value", 
               "frac_ks_sigs_value","avg_F1_diff_value", "avg_F1_diff_hout_value", 
               "nnaa_value")
priv_cols <- c("priv_loss_nndr_value", "priv_loss_nnaa_value","hit_rate_value", 
               "avg_nndr_value", "eps_identif_risk_value", "priv_loss_eps_value",  
               "mia_recall_value", "att_discl_risk_value")
cols <- c(util_cols, priv_cols)


#rnk_results$util_score <- rnk_results |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, cls_F1_diff, cls_F1_diff_hout, nnaa)) |> rowMeans()

util_df <- bmk_results |> dplyr::select(dataset, Epochs, model, all_of(util_cols)) |>
  dplyr::mutate(avg_F1_diff_value = abs(avg_F1_diff_value),
                avg_F1_diff_hout_value = abs(avg_F1_diff_hout_value)) |>
  dplyr::mutate(mutual_inf_diff = 1 - tanh(mutual_inf_diff_value),
                ks_tvd_stat = 1 - ks_tvd_stat_value,
                frac_ks_sigs = 1 - frac_ks_sigs_value,
                avg_F1_diff = 1 - abs(avg_F1_diff_value),
                avg_F1_diff_hout = 1 - abs(avg_F1_diff_hout_value),
                nnaa = 1 - nnaa_value) 
util_df$util_score <- util_df |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, avg_F1_diff, avg_F1_diff_hout, nnaa)) |> rowMeans()

priv_df <- bmk_results |> dplyr::select(dataset, Epochs, model, all_of(priv_cols)) |>
  dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                priv_loss_eps = 1 - abs(priv_loss_eps_value),
                hit_rate = 1 - hit_rate_value,
                eps_identif_risk = 1 - eps_identif_risk_value,
                mia_recall = 1 - mia_recall_value,
                att_discl_risk = 1 - att_discl_risk_value) 
priv_df$priv_score <- priv_df |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, mia_recall, att_discl_risk)) |> rowMeans()

up_df <- util_df |> dplyr::select(c(dataset, Epochs, model, util_score)) |> 
  dplyr::inner_join(priv_df |> dplyr::select(c(dataset, Epochs, model, priv_score)), by = c("dataset", "model", "Epochs"))


#colorpal <- paletteer_d("MetBrewer::Benedictus")[c(1,3,5,9,11,13)]
up_plot <- ggplot(data = up_df, aes(x=util_score, y=priv_score, fill=Epochs)) +
  geom_point(pch=21, color = "black", size=2) +
  labs(title = "Utility vs Privacy score", x = "Utility", y = "Privacy") +
  scale_fill_paletteer_d("rcartocolor::BluYl") +
xlim(0, 1) +
ylim(0, 1)

#up_plot <- up_plot + geom_point(data = df_test, aes(fill="Test"))
plot(up_plot)
