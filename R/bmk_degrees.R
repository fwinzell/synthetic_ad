library(dplyr)
library(Rmisc)
library(ggplot2)
library(ggpubr)
library(stringr)
library(ggforce)
library(paletteer)
library(ggsci)


degrees <- c(2,3,4) #c(0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 25, 50, 100, NA)
#samples <- c(rep(100, length(epsilons)-length(which(is.na(epsilons)))), 18)
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/a4/bmk_new_deg", degrees, "_epszero.csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(degrees)) {
  df <- read.csv(file_paths[i])
  df$Degree <- degrees[i]
  df$samples <- 18 #samples[i]
  res_list[[i]] <- df
}

# Combine all data frames
bmk_results <- bind_rows(res_list)
rnk_results <- NA 


# Mean - SD - Confidence intervals
util_cols <- c("mutual_inf_diff_value",  "ks_tvd_stat_value", 
               "frac_ks_sigs_value","avg_F1_diff_value", "avg_F1_diff_hout_value", 
               "nnaa_value")
priv_cols <- c("priv_loss_nndr_value", "priv_loss_nnaa_value","hit_rate_value", 
               "avg_nndr_value", "eps_identif_risk_value", "priv_loss_eps_value",  
               "mia_recall_value", "att_discl_risk_value")
cols <- c(util_cols, priv_cols)


bmk_results$Degree <- as.factor(bmk_results$Degree)


util_df <- bmk_results |> dplyr::select(dataset, Degree, model, all_of(util_cols)) |>
  dplyr::mutate(avg_F1_diff_value = abs(avg_F1_diff_value),
                avg_F1_diff_hout_value = abs(avg_F1_diff_hout_value)) |>
  dplyr::mutate(mutual_inf_diff = 1 - tanh(mutual_inf_diff_value),
                ks_tvd_stat = 1 - ks_tvd_stat_value,
                frac_ks_sigs = 1 - frac_ks_sigs_value,
                avg_F1_diff = 1 - abs(avg_F1_diff_value),
                avg_F1_diff_hout = 1 - abs(avg_F1_diff_hout_value),
                nnaa = 1 - nnaa_value) 
util_df$util_score <- util_df |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, avg_F1_diff, avg_F1_diff_hout, nnaa)) |> rowMeans()

priv_df <- bmk_results |> dplyr::select(dataset, Degree, model, all_of(priv_cols)) |>
  dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                priv_loss_eps = 1 - abs(priv_loss_eps_value),
                hit_rate = 1 - hit_rate_value,
                eps_identif_risk = 1 - eps_identif_risk_value,
                mia_recall = 1 - mia_recall_value,
                att_discl_risk = 1 - att_discl_risk_value) 
priv_df$priv_score <- priv_df |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, mia_recall, att_discl_risk)) |> rowMeans()

up_df <- util_df |> dplyr::select(c(dataset, Degree, model, util_score)) |> 
  dplyr::inner_join(priv_df |> dplyr::select(c(dataset, Degree, model, priv_score)), by = c("dataset", "model", "Degree"))


up_plot_deg <- ggplot(data = up_df, aes(x=util_score, y=priv_score, fill=Degree)) +
  geom_point(pch=21, color = "black", size=3) +
  labs(title = "Utility vs Privacy score", x = "Utility", y = "Privacy") +
  scale_fill_paletteer_d("ggsci::lanonc_lancet") +
  theme_minimal() +
  xlim(0.3, 0.8) + ylim(0.5, 1)

#up_plot <- up_plot + geom_point(data = df_test, aes(fill="Test"))
plot(up_plot_deg)

ggsave("~/R/DDLS-plots/up_plot_degrees.png", up_plot_deg, width = 6, height = 4, units = "in", dpi = 500)


up_df %>% dplyr::group_by(Degree) %>% dplyr::summarise(Mean_util = mean(util_score), 
                                                       SD_util  = sd(util_score),
                                                       Mean_priv = mean(priv_score),
                                                       SD_priv = sd(priv_score),
                                                       N = n()) %>% mutate(uci_lower = Mean_util - qt(0.975, df = N - 1) * SD_util / sqrt(N),
                                                                           uci_upper = Mean_util + qt(0.975, df = N - 1) * SD_util / sqrt(N),
                                                                           pci_lower = Mean_priv - qt(0.975, df = N - 1) * SD_priv / sqrt(N),
                                                                           pci_upper = Mean_priv + qt(0.975, df = N - 1) * SD_priv / sqrt(N),) %>% print()


##### FULL epsilon degree 2 vs 3 #######


epsilons <- c(5, 10, 25, 50, 100, 200, NA) 
samples <- c(rep(100, length(epsilons)-length(which(is.na(epsilons)))), 18)
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/bmk_new_deg3_eps", ifelse(is.na(epsilons), "zero", epsilons), ".csv")

res_list <- list()
for (i in seq_along(epsilons)) {
  df <- read.csv(file_paths[i])
  df$Epsilon <- epsilons[i]
  df$samples <- samples[i]
  res_list[[i]] <- df
}

bmk_deg_3 <- bind_rows(res_list)

file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/bmk_new_deg2_eps", ifelse(is.na(epsilons), "zero", epsilons), ".csv")
res_list <- list()
for (i in seq_along(epsilons)) {
  df <- read.csv(file_paths[i])
  df$Epsilon <- epsilons[i]
  df$samples <- samples[i]
  res_list[[i]] <- df
}

bmk_deg_2 <- bind_rows(res_list)


util_cols <- c("mutual_inf_diff_value",  "ks_tvd_stat_value", 
               "frac_ks_sigs_value","avg_F1_diff_value", "avg_F1_diff_hout_value", 
               "nnaa_value")
priv_cols <- c("priv_loss_nndr_value", "priv_loss_nnaa_value","hit_rate_value", 
               "avg_nndr_value", "eps_identif_risk_value", "priv_loss_eps_value",  
               "mia_recall_value", "att_discl_risk_value")
cols <- c(util_cols, priv_cols)

params <- expand.grid(eps = c(epsilons[1:6], 0), degs = c(2,3))
label_names <- sprintf("DS (deg: %s eps: %s)", params$degs, params$eps)

bmk_deg_3$Epsilon <- bmk_deg_3 %>% 
  select(Epsilon) %>% 
  mutate_all(~replace(., is.na(.), 0)) %>% 
  mutate(Epsilon = factor(Epsilon, levels = as.character(unique(Epsilon)))) %>% 
  pull(Epsilon)

bmk_deg_2$Epsilon <- bmk_deg_3 %>% 
  select(Epsilon) %>% 
  mutate_all(~replace(., is.na(.), 0)) %>% 
  mutate(Epsilon = factor(Epsilon, levels = as.character(unique(Epsilon)))) %>% 
  pull(Epsilon)  


util_df <- rbind(select(bmk_deg_2, dataset, model, all_of(util_cols)),
                 select(bmk_deg_3, dataset, model, all_of(util_cols))) |>
  dplyr::mutate(avg_F1_diff_value = abs(avg_F1_diff_value),
                avg_F1_diff_hout_value = abs(avg_F1_diff_hout_value)) |>
  dplyr::mutate(mutual_inf_diff = 1 - tanh(mutual_inf_diff_value),
                ks_tvd_stat = 1 - ks_tvd_stat_value,
                frac_ks_sigs = 1 - frac_ks_sigs_value,
                avg_F1_diff = 1 - abs(avg_F1_diff_value),
                avg_F1_diff_hout = 1 - abs(avg_F1_diff_hout_value),
                nnaa = 1 - nnaa_value) 
util_df$util_score <- util_df |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, avg_F1_diff, avg_F1_diff_hout, nnaa)) |> rowMeans()


util_df$Method <- factor(util_df$dataset, levels = unique(util_df$dataset), labels = label_names)

priv_df <- rbind(select(bmk_deg_2, dataset, model, all_of(priv_cols)),
                 select(bmk_deg_3, dataset, model, all_of(priv_cols))) |>
  dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                priv_loss_eps = 1 - abs(priv_loss_eps_value),
                hit_rate = 1 - hit_rate_value,
                eps_identif_risk = 1 - eps_identif_risk_value,
                mia_recall = 1 - mia_recall_value,
                att_discl_risk = 1 - att_discl_risk_value) 
priv_df$priv_score <- priv_df |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, att_discl_risk)) |> rowMeans()

priv_df$Method <- factor(priv_df$dataset, levels = unique(priv_df$dataset), labels=label_names)

up_df <- util_df |> dplyr::select(c(Method, model, util_score)) |> 
  dplyr::inner_join(priv_df |> dplyr::select(c(Method, model, priv_score)), by = c("Method", "model"))


#up_df$Method <- factor(up_df$dataset, levels = unique(up_df$dataset),
#                       labels=c("DS (eps=5)", "DS (eps=10)", "DS (eps=25)","DS (eps=50)","DS (eps=100)", 
#                                "DS (eps=200)", "DS w/o DP", "CTGAN (10 epochs)", 
#                                "CTGAN (50 epochs)", "CTGAN (100 epochs)", "Synthpop"))

colormap <- c(
  paletteer_d("rcartocolor::BluYl"),
  paletteer_d("ggsci::red_material")
)

up_plot_eps <- ggplot(data = up_df, aes(x=util_score, y=priv_score, fill=Method)) +
  geom_point(pch=21, color = "black", size=3) +
  labs(title = "Utility vs Privacy score", x = "Utility", y = "Privacy") +
  #scale_fill_paletteer_d("rcartocolor::BluYl") +
  scale_fill_manual(values = colormap) +
  theme_minimal() +
  xlim(0.3, 0.8) + ylim(0.5, 1)

plot(up_plot_eps)

ggsave("~/R/DDLS-plots/up_plot_deg_eps.png", up_plot_eps, width = 6, height = 4, units = "in", dpi = 500)


library(ggpubr)


# Extract legends
legend_deg <- get_legend(up_plot_deg)
legend_eps <- get_legend(up_plot_eps)

# Remove legends from main plots
pdeg_clean <- up_plot_deg + theme(legend.position = "none")
peps_clean <- up_plot_eps + theme(legend.position = "none")

# Arrange plots with equal width
plots <- ggarrange(pdeg_clean, peps_clean, ncol = 2, widths = c(1, 1))

# Put legends below (or to the right)
pp <- ggarrange(
  pdeg_clean, as_ggplot(legend_deg),
  peps_clean, as_ggplot(legend_eps),
  ncol = 4,
  widths = c(1, 0.18, 1, 0.5)   # adjust legend widths as needed
)
plot(pp)

ggsave("~/R/DDLS-plots/up_plot_degree_eps_comb.png", pp, width = 10, height = 4, units = "in", dpi = 500)

