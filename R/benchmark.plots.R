library(dplyr)
library(Rmisc)
library(ggplot2)
library(ggpubr)
library(stringr)
library(ggforce)
library(paletteer)
library(ggsci)

##### ADNI #####
epsilons <- c(5, 10, 50, 100, 200, NA) 
samples <- c(rep(100, length(epsilons)-length(which(is.na(epsilons)))), 18)
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/adni/bmk_new_deg2_eps", ifelse(is.na(epsilons), "zero", epsilons), ".csv")

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

# CTGAN 
epochs <- c(750) 
setting <- c("default", "optim")
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/adni/bmk_new_ctgan_", setting, "_epochs_", epochs, ".csv")
#file_paths <- c(file_paths, "~/Python/WASP-DDLS/SE-benchmark/bmk_ctgan_epochs_100.csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(file_paths)) {
  df <- read.csv(file_paths[i])
  df$Epochs <- epochs[i]
  df$samples <- 100
  res_list[[i]] <- df
}

# Combine all data frames
bmk_ctgan <- bind_rows(res_list)

bmk_ctgan$Epochs <- as.factor(bmk_ctgan$Epochs)


# Synthpop 
bmk_synthpop <- read.csv("~/Python/WASP-DDLS/SE-benchmark/adni/bmk_synthpop_2.csv") %>% mutate(
  Epochs = 0,
  samples = 100
)

# TabPFN
temps <- c('1.0', '0.75', '0.5', '0.25') 
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/adni/bmk_new_tabpfn_t_", temps, ".csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(file_paths)) {
  df <- read.csv(file_paths[i])
  df$temp <- temps[i]
  df$samples <- 100
  res_list[[i]] <- df
}

# Combine all data frames
bmk_tabpfn <- bind_rows(res_list)

bmk_tabpfn$temp <- as.factor(bmk_tabpfn$temp)


label_list = c("DS (5)", "DS (10)","DS (50)","DS (100)", 
               "DS (200)", "DS w/o DP", "CTGAN (default)", "CTGAN (optimal)",
               "Synthpop", 
               "TabPFN (t=1.0)", "TabPFN (t=0.75)", "TabPFN (t=0.5)", "TabPFN (t=0.25)")

util_df_adni <- rbind(select(bmk_ds, dataset, model, all_of(util_cols)),
                 select(bmk_ctgan, dataset, model, all_of(util_cols)),
                 select(bmk_synthpop, dataset, model, all_of(util_cols)),
                 select(bmk_tabpfn, dataset, model, all_of(util_cols))) |>
  dplyr::mutate(avg_F1_diff_value = abs(avg_F1_diff_value),
                avg_F1_diff_hout_value = abs(avg_F1_diff_hout_value)) |>
  dplyr::mutate(mutual_inf_diff = 1 - tanh(mutual_inf_diff_value),
                ks_tvd_stat = 1 - ks_tvd_stat_value,
                frac_ks_sigs = 1 - frac_ks_sigs_value,
                avg_F1_diff = 1 - abs(avg_F1_diff_value),
                avg_F1_diff_hout = 1 - abs(avg_F1_diff_hout_value),
                nnaa = 1 - nnaa_value) 
util_df_adni$util_score <- util_df_adni |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, avg_F1_diff, avg_F1_diff_hout, nnaa)) |> rowMeans()

util_df_adni$Method <- factor(util_df_adni$dataset, levels = unique(util_df_adni$dataset),
                       labels=label_list)

priv_df_adni <- rbind(select(bmk_ds, dataset, model, all_of(priv_cols)),
                 select(bmk_ctgan, dataset, model, all_of(priv_cols)),
                 select(bmk_synthpop, dataset, model, all_of(priv_cols)),
                 select(bmk_tabpfn, dataset, model, all_of(priv_cols))) |>
  dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                priv_loss_eps = 1 - abs(priv_loss_eps_value),
                hit_rate = 1 - hit_rate_value,
                eps_identif_risk = 1 - eps_identif_risk_value,
                mia_recall = 1 - mia_recall_value,
                att_discl_risk = 1 - att_discl_risk_value) 
priv_df_adni$priv_score <- priv_df_adni |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, att_discl_risk)) |> rowMeans()

priv_df_adni$Method <- factor(priv_df_adni$dataset, levels = unique(priv_df_adni$dataset),
                         labels=label_list)

up_df_adni <- util_df_adni |> dplyr::select(c(Method, model, util_score)) |> 
  dplyr::inner_join(priv_df_adni |> dplyr::select(c(Method, model, priv_score)), by = c("Method", "model"))


colormap <- c(
  paletteer_d("rcartocolor::BluYl")[1:6],
  paletteer_d("beyonce::X58")[3:4],
  "#663399",
  paletteer_d("ggsci::cyan_material")[c(1,3,5,7)]
)

up_plot_adni <- ggplot(data = up_df_adni, aes(x=util_score, y=priv_score, fill=Method)) +
  geom_point(pch=21, color = "black", size=3) +
  labs(title = "ADNI", x = "Utility", y = "Privacy") +
  #scale_fill_paletteer_d("rcartocolor::BluYl") +
  scale_fill_manual(values = colormap) +
  theme_minimal() +
  ylim(0.7,1.0) + xlim(0.3,0.85)

plot(up_plot_adni)

#ggsave("~/R/DDLS-plots/up_plot.png", up_plot, width = 7, height = 5, units = "in", dpi = 500)

###### A4 #########
epsilons <- c(5, 10, 50, 100, 200, NA) 
samples <- c(rep(100, length(epsilons)-length(which(is.na(epsilons)))), 23)
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/a4/bmk_new_deg2_eps", ifelse(is.na(epsilons), "zero", epsilons), ".csv")

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

# CTGAN 
epochs <- c(750) 
setting <- c("default", "optim")
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/a4/bmk_new_ctgan_", setting, "_epochs_", epochs, ".csv")
#file_paths <- c(file_paths, "~/Python/WASP-DDLS/SE-benchmark/bmk_ctgan_epochs_100.csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(file_paths)) {
  df <- read.csv(file_paths[i])
  df$Epochs <- epochs[i]
  df$samples <- 100
  res_list[[i]] <- df
}

# Combine all data frames
bmk_ctgan <- bind_rows(res_list)

bmk_ctgan$Epochs <- as.factor(bmk_ctgan$Epochs)


# Synthpop 
bmk_synthpop <- read.csv("~/Python/WASP-DDLS/SE-benchmark/a4/bmk_new_synthpop.csv") %>% mutate(
  Epochs = 0,
  samples = 100
)

# TabPFN 
temps <- c('1.0') 
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/a4/bmk_new_tabpfn_t_", temps, ".csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(file_paths)) {
  df <- read.csv(file_paths[i])
  df$temp <- temps[i]
  df$samples <- 100
  res_list[[i]] <- df
}

# Combine all data frames
bmk_tabpfn <- bind_rows(res_list)

bmk_tabpfn$temp <- as.factor(bmk_tabpfn$temp)


label_list = c("DS (5)", "DS (10)", "DS (50)", "DS (100)", "DS (200)", "DS w/o DP", 
               "CTGAN (default)", "CTGAN (optimal)", 
               "Synthpop", "TabPFN (t=1.0")

util_df_a4 <- rbind(select(bmk_ds, dataset, model, all_of(util_cols)),
                 select(bmk_ctgan, dataset, model, all_of(util_cols)),
                 select(bmk_synthpop, dataset, model, all_of(util_cols)),
                 select(bmk_tabpfn, dataset, model, all_of(util_cols))
                 ) |>
  dplyr::mutate(avg_F1_diff_value = abs(avg_F1_diff_value),
                avg_F1_diff_hout_value = abs(avg_F1_diff_hout_value)) |>
  dplyr::mutate(mutual_inf_diff = 1 - tanh(mutual_inf_diff_value),
                ks_tvd_stat = 1 - ks_tvd_stat_value,
                frac_ks_sigs = 1 - frac_ks_sigs_value,
                avg_F1_diff = 1 - abs(avg_F1_diff_value),
                avg_F1_diff_hout = 1 - abs(avg_F1_diff_hout_value),
                nnaa = 1 - nnaa_value) 
util_df_a4$util_score <- util_df_a4 |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, avg_F1_diff, avg_F1_diff_hout, nnaa)) |> rowMeans()

util_df_a4$Method <- factor(util_df_a4$dataset, levels = unique(util_df_a4$dataset),
                         labels=label_list)

priv_df_a4 <- rbind(select(bmk_ds, dataset, model, all_of(priv_cols)),
                 select(bmk_ctgan, dataset, model, all_of(priv_cols)),
                 select(bmk_synthpop, dataset, model, all_of(priv_cols)),
                 select(bmk_tabpfn, dataset, model, all_of(priv_cols))
                 ) |>
  dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                priv_loss_eps = 1 - abs(priv_loss_eps_value),
                hit_rate = 1 - hit_rate_value,
                eps_identif_risk = 1 - eps_identif_risk_value,
                mia_recall = 1 - mia_recall_value,
                att_discl_risk = 1 - att_discl_risk_value) 
priv_df_a4$priv_score <- priv_df_a4 |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, att_discl_risk)) |> rowMeans()

priv_df_a4$Method <- factor(priv_df_a4$dataset, levels = unique(priv_df_a4$dataset),
                         labels=label_list)

up_df_a4 <- util_df_a4 |> dplyr::select(c(Method, model, util_score)) |> 
  dplyr::inner_join(priv_df_a4 |> dplyr::select(c(Method, model, priv_score)), by = c("Method", "model"))

up_df_a4 <- up_df_a4 %>% mutate(Framework = as.factor(gsub(" .*", "", Method))) 

#up_df$Method <- factor(up_df$dataset, levels = unique(up_df$dataset),
#                       labels=c("DS (eps=5)", "DS (eps=10)", "DS (eps=25)","DS (eps=50)","DS (eps=100)", 
#                                "DS (eps=200)", "DS w/o DP", "CTGAN (10 epochs)", 
#                                "CTGAN (50 epochs)", "CTGAN (100 epochs)", "Synthpop"))

colormap <- c(
  paletteer_d("rcartocolor::BluYl")[1:6],
  paletteer_d("beyonce::X58")[3:4],
  "#663399",
  paletteer_d("ggsci::cyan_material")[c(1,3,5,7)]
)

up_plot_a4 <- ggplot(data = up_df_a4, aes(x=util_score, y=priv_score, fill=Method)) +
  geom_point(pch=21, color="black", size=3) +
  labs(title = "A4", x = "Utility", y = "Privacy") +
  #scale_fill_paletteer_d("rcartocolor::BluYl") +
  scale_fill_manual(values = colormap) +
  theme_minimal() +
  ylim(0.7,1.0) + xlim(0.3,0.85)

#plot(up_plot_a4)

up_plot <- ggarrange(up_plot_adni, up_plot_a4, nrow=2, common.legend = TRUE, legend = "right")
plot(up_plot)

#ggsave("~/R/DDLS-plots/up_plot.jpg", up_plot, width = 7, height = 10, units = "in", dpi = 1000)

###### GRIDS OF METRICS ######
library(lme4)

# Privacy grid

privs <- c("priv_loss_nndr_value", "priv_loss_nnaa_value", "eps_identif_risk_value", "hit_rate_value", "mia_recall_value", "att_discl_risk_value")
priv_titels <- c(
  priv_loss_nndr_value = "NNDR privacy loss",
  priv_loss_nnaa_value = "NNAA privacy loss",
  eps_identif_risk_value = "Epsilon identifiability",
  hit_rate_value = "Hitting rate",
  mia_recall_value = "MIA",
  att_discl_risk_value = "ADR"
)
priv_df_adni <- priv_df_adni %>% mutate(eps_identif_risk_value = eps_identif_risk_value*100, 
                   hit_rate_value = hit_rate_value*100)
priv_df_a4 <- priv_df_a4 %>% mutate(eps_identif_risk_value = eps_identif_risk_value*100, 
                                        hit_rate_value = hit_rate_value*100)

get_privacy_plot <- function(priv_df, p) {
  pp <- ggplot(data = priv_df, aes(x = Method, y = !!sym(p), fill = Method)) +
    labs(title = priv_titels[p],
         x = "",
         y = "") +
    scale_fill_manual(values = colormap) +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 30, vjust = 1, hjust=1, size = 8))
  if(p %in% c("eps_identif_risk_value", "hit_rate_value")) {
    yval = 9.0
    pp <- pp + geom_hline(yintercept = yval, linetype = "dashed", color = "darkred") +
      geom_rect(
        xmin = -Inf, xmax = Inf,
        ymin = yval, ymax = Inf,
        fill = "grey80",
        alpha = 0.5
      )
  } else if(p %in% c("priv_loss_nndr_value", "priv_loss_nnaa_value")) {
    mu = mean(priv_df[[p]], na.rm=T)
    #m = length(unique(priv_df$Method))
    #n = nrow(priv_df)/m
    #se = sd(priv_df[[p]], na.rm=T)/sqrt(n)
    #t_crit <- qt(1 - 0.01 / (2 * m), df = round(n)-1)
    
    form <- as.formula(paste(p, "~ 1 + (1 | Method)"))
    fit <- lmer(form, data = priv_df)
    cis <- confint(fit, level = 0.99, parm = "(Intercept)")
    
    upper = cis[2]
    lower = cis[1]
    
    pp <- pp + 
      geom_hline(yintercept = mu, linetype = "solid", color = "lightgray") +
      geom_hline(yintercept = upper, linetype = "dashed", color = "darkred", alpha=0.5) +
      geom_hline(yintercept = lower, linetype = "dashed", color = "darkred", alpha=0.5) +
      geom_rect(xmin = -Inf, xmax = Inf, ymin = upper, ymax = Inf, fill = "grey80", alpha = 0.2) +
      geom_rect(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = lower, fill = "grey80", alpha = 0.2)
  } else if(p %in% c("mia_recall_value", "att_discl_risk_value")) {
    yval = 0.5
    pp <- pp + geom_hline(yintercept = yval, linetype = "dashed", color = "darkred") +
      geom_rect(
        xmin = -Inf, xmax = Inf,
        ymin = yval, ymax = Inf,
        fill = "grey80",
        alpha = 0.5
      )
  }
  pp <- pp + geom_hline(yintercept = 0.0, linetype = "dotted", color = "black") +
    geom_boxplot()
  
  return(pp)
}


pPlotlist_adni <- list()
pPlotlist_a4 <- list()
for (p in privs) {
  pp_adni <- get_privacy_plot(priv_df_adni, p)
  pp_a4 <- get_privacy_plot(priv_df_a4, p)
  pPlotlist_adni[[paste0(p, "_adni")]] <- pp_adni
  pPlotlist_a4[[paste0(p, "_a4")]] <- pp_a4
}

priv_grid_adni <- ggarrange(plotlist = pPlotlist_adni, ncol = 2, nrow = 3, common.legend = TRUE, legend = "right")
plot(priv_grid_adni)

priv_grid_a4 <- ggarrange(plotlist = pPlotlist_a4, ncol = 2, nrow = 3, common.legend = TRUE, legend = "right")
plot(priv_grid_a4)

ggsave("~/R/DDLS-plots/priv_grid_adni.tiff", priv_grid_adni, width = 10, height = 10, units = "in", dpi = 500)
ggsave("~/R/DDLS-plots/priv_grid_a4.tiff", priv_grid_a4, width = 10, height = 10, units = "in", dpi = 500)

#ggsave("~/R/DDLS-plots/util_grid.png", util_grid, width = 10, height = 10, units = "in", dpi = 500)
#ggsave("~/R/DDLS-plots/priv_grid.png", priv_grid, width = 10, height = 10, units = "in", dpi = 500)

utils <- c("mutual_inf_diff_value", "ks_tvd_stat_value", "frac_ks_sigs_value", "avg_F1_diff_value", "avg_F1_diff_hout_value", "nnaa_value")
# Mapping column names
util_titles <- c(
  mutual_inf_diff_value = "MI diff",
  ks_tvd_stat_value = "KS/TVD stat",
  frac_ks_sigs_value = "Frac. KS sigs",
  avg_F1_diff_value = "Avg. Diff. F1",
  avg_F1_diff_hout_value = "Avg. Diff. F1 (test)",
  nnaa_value = "NNAA value"
)

get_util_plot <- function(util_df, u) {
  up <- ggplot(data = util_df, aes(x = Method, y = !!sym(u), fill = Method)) +
    labs(title = util_titles[u],
         x = "",
         y = "") + 
    scale_fill_manual(values = colormap) +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 30, vjust = 1, hjust=1, size = 8)) 
  if (u == 'nnaa_value') {
    mu = mean(util_df[[u]], na.rm=T)
    #m = length(unique(priv_df$Method))
    #n = nrow(priv_df)/m
    #se = sd(priv_df[[p]], na.rm=T)/sqrt(n)
    #t_crit <- qt(1 - 0.01 / (2 * m), df = round(n)-1)
    
    form <- as.formula(paste(u, "~ 1 + (1 | Method)"))
    fit <- lmer(form, data = util_df)
    cis <- confint(fit, level = 0.99, parm = "(Intercept)")
    
    upper = cis[2]
    lower = cis[1]
    
    up <- up + 
      geom_hline(yintercept = mu, linetype = "solid", color = "lightgray") +
      geom_hline(yintercept = upper, linetype = "dashed", color = "darkred", alpha=0.5) +
      geom_hline(yintercept = lower, linetype = "dashed", color = "darkred", alpha=0.5) +
      geom_rect(xmin = -Inf, xmax = Inf, ymin = upper, ymax = Inf, fill = "grey80", alpha = 0.2) +
      geom_rect(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = lower, fill = "grey80", alpha = 0.2)
  } else if (u == "frac_ks_sigs_value"){
    up <- up + geom_hline(yintercept = 1.0, linetype = "solid", color = "gray")
  } 
  up <- up + geom_boxplot() 
  return(up)
}



uPlotlist_adni <- list()
uPlotlist_a4 <- list()
for (u in utils) {
  uPlotlist_adni[[u]] <- get_util_plot(util_df_adni, u)
  uPlotlist_a4[[u]] <- get_util_plot(util_df_a4, u)
}

util_grid_adni <- ggarrange(plotlist = uPlotlist_adni, ncol = 2, nrow = 3, common.legend = TRUE, legend = "right")
plot(util_grid_adni)

util_grid_a4 <- ggarrange(plotlist = uPlotlist_a4, ncol = 2, nrow = 3, common.legend = TRUE, legend = "right")
plot(util_grid_a4)

ggsave("~/R/DDLS-plots/util_grid_adni.tiff", util_grid_adni, width = 10, height = 10, units = "in", dpi = 500)
ggsave("~/R/DDLS-plots/util_grid_a4.tiff", util_grid_a4, width = 10, height = 10, units = "in", dpi = 500)

##### Get specific values #####

summary_stats_adni <- up_df_adni %>%
  group_by(Method) %>%
  dplyr::summarize(
    mean_u = mean(util_score),
    mean_p = mean(priv_score),
    sd_u = sd(util_score),
    sd_p = sd(priv_score),
    num = n(),
    ci_margin_u = qt(0.975, df = n() - 1) * sd(util_score) / sqrt(n()),
    ci_margin_p = qt(0.975, df = n() - 1) * sd(priv_score) / sqrt(n()),
    ci_lwr_u = mean_u - ci_margin_u,
    ci_upr_u = mean_u + ci_margin_u,
    ci_lwr_p = mean_p - ci_margin_p,
    ci_upr_p = mean_p + ci_margin_p
  )

summary_stats_a4 <- up_df_a4 %>%
  group_by(Method) %>%
  dplyr::summarize(
    mean_u = mean(util_score),
    mean_p = mean(priv_score),
    ci_margin_u = qt(0.975, df = n() - 1) * sd(util_score) / sqrt(n()),
    ci_margin_p = qt(0.975, df = n() - 1) * sd(priv_score) / sqrt(n()),
    ci_lwr_u = mean_u - ci_margin_u,
    ci_upr_u = mean_u + ci_margin_u,
    ci_lwr_p = mean_p - ci_margin_p,
    ci_upr_p = mean_p + ci_margin_p
  )

priv_df_adni %>% group_by(dataset) %>% 
  dplyr::summarise(Mean = mean(eps_identif_risk_value),
                   SD = sd(eps_identif_risk_value),
                   N = n(),
                   margin = qt(0.975, df = n() - 1) * sd(eps_identif_risk_value) / sqrt(n()),
                   lower = Mean - margin,
                   upper = Mean + margin) %>% filter(upper > 0.09)

priv_df_a4 %>% group_by(dataset) %>% 
  dplyr::summarise(Mean = mean(eps_identif_risk_value),
                   SD = sd(eps_identif_risk_value),
                   N = n(),
                   margin = qt(0.975, df = n() - 1) * sd(eps_identif_risk_value) / sqrt(n()),
                   lower = Mean - margin,
                   upper = Mean + margin) %>% filter(upper > 0.09)
                                                 

priv_df_adni %>% group_by(dataset) %>% 
  dplyr::summarise(Mean = mean(att_discl_risk_value),
                   SD = sd(att_discl_risk_value),
                   N = n(),
                   margin = qt(0.975, df = n() - 1) * sd(att_discl_risk_value) / sqrt(n()),
                   lower = Mean - margin,
                   upper = Mean + margin) %>% filter(upper > 0.1)


priv_df_a4 %>% group_by(dataset) %>% 
  dplyr::summarise(Mean = mean(att_discl_risk_value),
                   SD = sd(att_discl_risk_value),
                   N = n(),
                   margin = qt(0.975, df = n() - 1) * sd(att_discl_risk_value) / sqrt(n()),
                   lower = Mean - margin,
                   upper = Mean + margin) %>% filter(upper > 0.1)

priv_df_adni %>% group_by(dataset) %>% 
  dplyr::summarise(Mean = mean(mia_recall_value),
                   SD = sd(mia_recall_value),
                   N = n(),
                   margin = qt(0.975, df = n() - 1) * sd(mia_recall_value) / sqrt(n()),
                   lower = Mean - margin,
                   upper = Mean + margin) %>% filter(upper > 0.1)

priv_df_a4 %>% group_by(dataset) %>% 
  dplyr::summarise(Mean = mean(mia_recall_value),
                   SD = sd(mia_recall_value),
                   N = n(),
                   margin = qt(0.975, df = n() - 1) * sd(mia_recall_value) / sqrt(n()),
                   lower = Mean - margin,
                   upper = Mean + margin) %>% filter(upper > 0.1)

# Bonus experiment
#### ADNI+ ####
epsilons <- c(100, NA) 
samples <- c(100, 18)
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/adni_plus/bmk_new_deg2_eps", ifelse(is.na(epsilons), "zero", epsilons), ".csv")

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

# CTGAN 
epochs <- c(750) 
setting <- c("default")
file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/adni_plus/bmk_new_ctgan_", setting, "_epochs_", epochs, ".csv")
#file_paths <- c(file_paths, "~/Python/WASP-DDLS/SE-benchmark/bmk_ctgan_epochs_100.csv")

# Read CSVs in a loop
res_list <- list()
for (i in seq_along(file_paths)) {
  df <- read.csv(file_paths[i])
  df$Epochs <- epochs[i]
  df$samples <- 100
  res_list[[i]] <- df
}

# Combine all data frames
bmk_ctgan <- bind_rows(res_list)

bmk_ctgan$Epochs <- as.factor(bmk_ctgan$Epochs)


# Synthpop 
bmk_synthpop <- read.csv("~/Python/WASP-DDLS/SE-benchmark/adni_plus/bmk_new_synthpop.csv") %>% mutate(
  Epochs = 0,
  samples = 100
)

# TabPFN
bmk_tabpfn <- read.csv("~/Python/WASP-DDLS/SE-benchmark/adni_plus/bmk_new_tabpfn_t_1.0.csv")


label_list = c("DS (100)", "DS w/o DP", "CTGAN (default)", "Synthpop", "TabPFN (t=1.0)") 
               

util_df_adni_plus <- rbind(select(bmk_ds, dataset, model, all_of(util_cols)),
                      select(bmk_ctgan, dataset, model, all_of(util_cols)),
                      select(bmk_synthpop, dataset, model, all_of(util_cols)),
                      select(bmk_tabpfn, dataset, model, all_of(util_cols))
                      ) |>
  dplyr::mutate(avg_F1_diff_value = abs(avg_F1_diff_value),
                avg_F1_diff_hout_value = abs(avg_F1_diff_hout_value)) |>
  dplyr::mutate(mutual_inf_diff = 1 - tanh(mutual_inf_diff_value),
                ks_tvd_stat = 1 - ks_tvd_stat_value,
                frac_ks_sigs = 1 - frac_ks_sigs_value,
                avg_F1_diff = 1 - abs(avg_F1_diff_value),
                avg_F1_diff_hout = 1 - abs(avg_F1_diff_hout_value),
                nnaa = 1 - nnaa_value) 
util_df_adni_plus$util_score <- util_df_adni_plus |> dplyr::select(c(mutual_inf_diff, ks_tvd_stat, frac_ks_sigs, avg_F1_diff, avg_F1_diff_hout, nnaa)) |> rowMeans()

util_df_adni_plus$Method <- factor(util_df_adni_plus$dataset, levels = unique(util_df_adni_plus$dataset),
                              labels=label_list)

priv_df_adni_plus <- rbind(select(bmk_ds, dataset, model, all_of(priv_cols)),
                      select(bmk_ctgan, dataset, model, all_of(priv_cols)),
                      select(bmk_synthpop, dataset, model, all_of(priv_cols)),
                      select(bmk_tabpfn, dataset, model, all_of(priv_cols))
                      ) |>
  dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                priv_loss_eps = 1 - abs(priv_loss_eps_value),
                hit_rate = 1 - hit_rate_value,
                eps_identif_risk = 1 - eps_identif_risk_value,
                mia_recall = 1 - mia_recall_value,
                att_discl_risk = 1 - att_discl_risk_value) 
priv_df_adni_plus$priv_score <- priv_df_adni_plus |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, att_discl_risk)) |> rowMeans()

priv_df_adni_plus$Method <- factor(priv_df_adni_plus$dataset, levels = unique(priv_df_adni_plus$dataset),
                              labels=label_list)

up_df_adni_plus <- util_df_adni_plus |> dplyr::select(c(Method, model, util_score)) |> 
  dplyr::inner_join(priv_df_adni_plus |> dplyr::select(c(Method, model, priv_score)), by = c("Method", "model"))


colormap <- c(
  paletteer_d("rcartocolor::BluYl")[1:2],
  paletteer_d("beyonce::X58")[3],
  "#663399",
  paletteer_d("ggsci::cyan_material")[c(1,3,5,7)]
)

up_plot_adni_plus <- ggplot(data = up_df_adni_plus, aes(x=util_score, y=priv_score, fill=Method)) +
  labs(title = "ADNI+", x = "Utility", y = "Privacy") +
  #scale_fill_paletteer_d("rcartocolor::BluYl") +
  scale_fill_manual(values = colormap) +
  theme_minimal() 
  #ylim(0.7,1.0) + xlim(0.3,0.85)

up_df_adni_ <- up_df_adni %>% filter(Method %in% up_df_adni_plus$Method)
up_plot_adni_plus <- up_plot_adni_plus + 
  geom_point(pch=21, color="gray", size = 3, data=up_df_adni_, alpha=0.75) + 
  geom_point(pch=21, color = "black", size=3) 

plot(up_plot_adni_plus)

up_plot_adni_plus <- up_plot_adni_plus + theme(plot.margin = margin(5.5, 125, 5.5, 125))

up_plot_all <- ggarrange(up_plot_adni, up_plot_a4, ncol=2, common.legend = TRUE, legend = "right")
up_plot_all <- ggarrange(up_plot_all, up_plot_adni_plus, nrow=2, common.legend = FALSE)
plot(up_plot_all)

##### Saving and stuff ####
ggsave("~/R/DDLS-plots/up_plot_all.tiff", up_plot_all, width = 10, height = 9, units = "in", dpi = 500)


summary_stats_adni_plus <- up_df_adni_plus %>%
  group_by(Method) %>%
  dplyr::summarize(
    mean_u = mean(util_score),
    mean_p = mean(priv_score),
    sd_u = sd(util_score),
    sd_p = sd(priv_score),
    num = n(),
    ci_margin_u = qt(0.975, df = n() - 1) * sd(util_score) / sqrt(n()),
    ci_margin_p = qt(0.975, df = n() - 1) * sd(priv_score) / sqrt(n()),
    ci_lwr_u = mean_u - ci_margin_u,
    ci_upr_u = mean_u + ci_margin_u,
    ci_lwr_p = mean_p - ci_margin_p,
    ci_upr_p = mean_p + ci_margin_p
  )


res <- summary_stats_adni %>% inner_join(summary_stats_adni_plus, by='Method', suffix = c('.adni', '.adni_plus')) %>%
  mutate(ci.bound_u = (mean_u.adni - mean_u.adni_plus) - 1.96 * sqrt((sd_u.adni^2/num.adni) + (sd_u.adni^2/num.adni)), 
         ci.bound_p = (mean_p.adni - mean_p.adni_plus) - 1.96 * sqrt((sd_p.adni^2/num.adni) + (sd_p.adni^2/num.adni)))

res <- data.frame()
for(method in unique(up_df_adni_plus$Method)) {
  a <- up_df_adni_plus %>% filter(Method == method)
  b <- up_df_adni %>% filter(Method == method)
  
  t_u <- t.test(a$util_score, b$util_score, alternative = "two.sided", var.equal=F)
  t_p <- t.test(a$priv_score, b$priv_score, alternative = "two.sided", var.equal=F)
  res <- rbind(res, data.frame(Method = method,
                               #t_stat_u = t_u$statistic,
                               p_value_u = t_u$p.value * nrow(a),
                               mean_u_x = t_u$estimate[1],
                               mean_u_y = t_u$estimate[2],
                               #t_stat_p = t_p$statistic,
                               p_value_p = t_p$p.value * nrow(a),
                               mean_p_x = t_p$estimate[1],
                               mean_p_y = t_p$estimate[2]
                               ))
}


