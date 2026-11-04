library(dplyr)
library(Rmisc)
library(ggplot2)
library(ggpubr)
library(stringr)
library(ggforce)
library(paletteer)
library(ggsci)

load_benchmark_data <- function(dataset,
                                #main_dir, # /path/to/dir_with_syntheval_output
                                epsilons_ds = c(5, 10, 50, 100, 200, NA), 
                                epochs_ctgan = 750, 
                                settings_ctgan = c("default", "optim"),
                                temps_tabpfn = c('1.0', '0.75', '0.5', '0.25'),
                                eps_td = c('1.0', '10.0', '25.0', '50.0', '100.0')) {
  
  samples <- c(rep(100, length(epsilons_ds)-length(which(is.na(epsilons_ds)))), 18)
  file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/", dataset, "/bmk_new_deg2_eps", ifelse(is.na(epsilons_ds), "zero", epsilons_ds), ".csv")
  
  # Read CSVs in a loop
  res_list <- list()
  for (i in seq_along(epsilons_ds)) {
    if (file.exists(file_paths[i])) {
    df <- read.csv(file_paths[i])
    df$Epsilon <- epsilons_ds[i]
    df$samples <- samples[i]
    res_list[[i]] <- df
    }
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
  #setting <- c("default", "optim")
  file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/", dataset, "/bmk_new_ctgan_", settings_ctgan, "_epochs_", epochs_ctgan, ".csv")
  #file_paths <- c(file_paths, "~/Python/WASP-DDLS/SE-benchmark/bmk_ctgan_epochs_100.csv")
  
  # Read CSVs in a loop
  res_list <- list()
  for (i in seq_along(file_paths)) {
    if (file.exists(file_paths[i])) {
    df <- read.csv(file_paths[i])
    df$Epochs <- epochs_ctgan
    df$samples <- 100
    res_list[[i]] <- df
    }
  }
  
  # Combine all data frames
  bmk_ctgan <- bind_rows(res_list)
  
  bmk_ctgan$Epochs <- as.factor(bmk_ctgan$Epochs)
  
  
  # Synthpop 
  bmk_synthpop <- read.csv(paste0("~/Python/WASP-DDLS/SE-benchmark/", dataset, "/bmk_new_synthpop.csv")) %>% mutate(
    Epochs = 0,
    samples = 100
  )
  
  # TabPFN
  file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/", dataset, "/bmk_new_tabpfn_t_", temps_tabpfn, ".csv")
  
  # Read CSVs in a loop
  res_list <- list()
  for (i in seq_along(file_paths)) {
    if (file.exists(file_paths[i])) {
      df <- read.csv(file_paths[i])
      df$temp <- temps_tabpfn[i]
      df$samples <- 100
      res_list[[i]] <- df 
    }
  }
  
  # Combine all data frames
  bmk_tabpfn <- bind_rows(res_list)
  
  bmk_tabpfn$temp <- as.factor(bmk_tabpfn$temp)
  
  
  # TableDiffusion
  file_paths <- paste0("~/Python/WASP-DDLS/SE-benchmark/", dataset, "/bmk_new_tablediffusion_td_epsilon_", eps_td, ".csv")
  
  # Read CSVs in a loop
  res_list <- list()
  for (i in seq_along(file_paths)) {
    df <- read.csv(file_paths[i])
    df$epsilon <- eps_td[i]
    df$samples <- 100
    res_list[[i]] <- df
  }
  
  # Combine all data frames
  bmk_td <- bind_rows(res_list)
  
  bmk_td$epsilon <- as.factor(bmk_td$epsilon)
  
  label_list = c(paste0("DS (", na.omit(epsilons_ds),  ")"), 
                 "DS w/o DP", 
                 paste0("CTGAN (", settings_ctgan, ")"),
                 "Synthpop", 
                 paste0("TabPFN (t=", unique(bmk_tabpfn$temp), ")"),
                 paste0("TD (", round(as.numeric(eps_td)), ")"))
  
  util_df <- rbind(select(bmk_ds, dataset, model, all_of(util_cols)),
                        select(bmk_ctgan, dataset, model, all_of(util_cols)),
                        select(bmk_synthpop, dataset, model, all_of(util_cols)),
                        select(bmk_tabpfn, dataset, model, all_of(util_cols)),
                        select(bmk_td, dataset, model, all_of(util_cols))
  ) |>
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
                                labels=label_list)
  
  priv_df <- rbind(select(bmk_ds, dataset, model, all_of(priv_cols)),
                        select(bmk_ctgan, dataset, model, all_of(priv_cols)),
                        select(bmk_synthpop, dataset, model, all_of(priv_cols)),
                        select(bmk_tabpfn, dataset, model, all_of(priv_cols)),
                        select(bmk_td, dataset, model, all_of(priv_cols))
  ) |>
    dplyr::mutate(priv_loss_nndr = 1 - abs(priv_loss_nndr_value),
                  priv_loss_nnaa = 1 - abs(priv_loss_nnaa_value),
                  priv_loss_eps = 1 - abs(priv_loss_eps_value),
                  hit_rate = 1 - hit_rate_value,
                  eps_identif_risk = 1 - eps_identif_risk_value,
                  mia_recall = 1 - mia_recall_value,
                  att_discl_risk = 1 - att_discl_risk_value) 
  priv_df$priv_score <- priv_df |> dplyr::select(c(priv_loss_nndr, priv_loss_nnaa, priv_loss_eps, hit_rate, eps_identif_risk, att_discl_risk)) |> rowMeans()
  
  priv_df$Method <- factor(priv_df$dataset, levels = unique(priv_df$dataset),
                                labels=label_list)
  
  up_df <- util_df |> dplyr::select(c(Method, model, util_score)) |> 
    dplyr::inner_join(priv_df |> dplyr::select(c(Method, model, priv_score)), by = c("Method", "model"))
  
  return(
    list(
      'util_df' = util_df,
      'priv_df' = priv_df,
      'up_df' = up_df
    )
  )
}

##### ADNI #####
adni_list <- load_benchmark_data('adni')


colormap_adni <- c(
  paletteer_d("rcartocolor::BluYl")[1:6],
  paletteer_d("beyonce::X58")[3:4],
  "#663399",
  paletteer_d("ggsci::cyan_material")[c(1,3,5,7)],
  paletteer_d("MetBrewer::Benedictus")[1:5]
)

up_plot_adni <- ggplot(data = adni_list$up_df, aes(x=util_score, y=priv_score, fill=Method)) +
  geom_point(pch=21, color = "black", size=3) +
  labs(title = "ADNI", x = "Utility", y = "Privacy") +
  #scale_fill_paletteer_d("rcartocolor::BluYl") +
  scale_fill_manual(values = colormap_adni) +
  theme_minimal() +
  ylim(0.7,1.0) + xlim(0.1,0.85)

plot(up_plot_adni)

###### A4 #########
a4_list <- load_benchmark_data('a4', temps_tabpfn = c('1.0'))

colormap_a4 <- c(
  paletteer_d("rcartocolor::BluYl")[1:6],
  paletteer_d("beyonce::X58")[3:4],
  "#663399",
  paletteer_d("ggsci::cyan_material")[c(1)],
  paletteer_d("MetBrewer::Benedictus")[1:5]
)

up_plot_a4 <- ggplot(data = a4_list$up_df, aes(x=util_score, y=priv_score, fill=Method)) +
  geom_point(pch=21, color="black", size=3) +
  labs(title = "A4", x = "Utility", y = "Privacy") +
  #scale_fill_paletteer_d("rcartocolor::BluYl") +
  scale_fill_manual(values = colormap_a4) +
  theme_minimal() +
  ylim(0.7,1.0) + xlim(0.1,0.85)

plot(up_plot_a4)

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
priv_df_adni <- adni_list$priv_df
priv_df_a4 <- a4_list$priv_df
priv_df_adni <- priv_df_adni %>% mutate(eps_identif_risk_value = eps_identif_risk_value*100, 
                   hit_rate_value = hit_rate_value*100)
priv_df_a4 <- priv_df_a4 %>% mutate(eps_identif_risk_value = eps_identif_risk_value*100, 
                                        hit_rate_value = hit_rate_value*100)

get_privacy_plot <- function(priv_df, p, colormap) {
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
  pp_adni <- get_privacy_plot(priv_df_adni, p, colormap_adni)
  pp_a4 <- get_privacy_plot(priv_df_a4, p, colormap_a4)
  pPlotlist_adni[[paste0(p, "_adni")]] <- pp_adni
  pPlotlist_a4[[paste0(p, "_a4")]] <- pp_a4
}

priv_grid_adni <- ggarrange(plotlist = pPlotlist_adni, ncol = 2, nrow = 3, common.legend = TRUE, legend = "right")
plot(priv_grid_adni)

priv_grid_a4 <- ggarrange(plotlist = pPlotlist_a4, ncol = 2, nrow = 3, common.legend = TRUE, legend = "right")
plot(priv_grid_a4)

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

get_util_plot <- function(util_df, u, colormap) {
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
  uPlotlist_adni[[u]] <- get_util_plot(adni_list$util_df, u, colormap_adni)
  uPlotlist_a4[[u]] <- get_util_plot(a4_list$util_df, u, colormap_a4)
}

util_grid_adni <- ggarrange(plotlist = uPlotlist_adni, ncol = 2, nrow = 3, common.legend = TRUE, legend = "right")
plot(util_grid_adni)

util_grid_a4 <- ggarrange(plotlist = uPlotlist_a4, ncol = 2, nrow = 3, common.legend = TRUE, legend = "right")
plot(util_grid_a4)

##### Get specific values #####

summary_stats_adni <- adni_list$up_df %>%
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

summary_stats_a4 <- a4_list$up_df %>%
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
adni_plus_list <- load_benchmark_data('adni_plus', 
                                      epsilons_ds = c(100, NA), 
                                      settings_ctgan = c("default"), 
                                      temps_tabpfn = c('1.0'),
                                      eps_td = c('10.0', '50.0') )

colormap_plus <- c(
  paletteer_d("rcartocolor::BluYl")[c(4,6)],
  paletteer_d("beyonce::X58")[3],
  "#663399",
  paletteer_d("ggsci::cyan_material")[c(1)],
  paletteer_d("MetBrewer::Benedictus")[c(2,4,5)]
)

up_plot_adni_plus <- ggplot(data = adni_plus_list$up_df, aes(x=util_score, y=priv_score, fill=Method)) +
  labs(title = "ADNI+", x = "Utility", y = "Privacy") +
  #scale_fill_paletteer_d("rcartocolor::BluYl") +
  scale_fill_manual(values = colormap_plus) +
  theme_minimal() +
  ylim(0.7,1.0) + xlim(0.1,0.85)

up_df_adni_ <- adni_list$up_df %>% filter(Method %in% adni_plus_list$up_df$Method)
up_plot_adni_plus <- up_plot_adni_plus + 
  geom_point(pch=21, color="gray", size = 3, data=up_df_adni_, alpha=0.75) + 
  geom_point(pch=21, color = "black", size=3) 

plot(up_plot_adni_plus)

up_plot_adni_plus <- up_plot_adni_plus + theme(plot.margin = margin(5.5, 125, 5.5, 125))

up_plot_all <- ggarrange(up_plot_adni, up_plot_a4, ncol=2, common.legend = TRUE, legend = "right")
up_plot_all <- ggarrange(up_plot_all, up_plot_adni_plus, nrow=2, common.legend = FALSE)
plot(up_plot_all)

##### Saving and stuff ####
ggsave("~/R/DDLS-plots/priv_grid_adni.tiff", priv_grid_adni, width = 12, height = 12, units = "in", dpi = 500)
ggsave("~/R/DDLS-plots/priv_grid_a4.tiff", priv_grid_a4, width = 12, height = 12, units = "in", dpi = 500)

ggsave("~/R/DDLS-plots/util_grid_adni.tiff", util_grid_adni, width = 12, height = 12, units = "in", dpi = 500)
ggsave("~/R/DDLS-plots/util_grid_a4.tiff", util_grid_a4, width = 12, height = 12, units = "in", dpi = 500)

ggsave("~/R/DDLS-plots/up_plot_all.tiff", up_plot_all, width = 12, height = 10, units = "in", dpi = 500)


summary_stats_adni_plus <- adni_plus_list$up_df %>%
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
for(method in unique(adni_plus_list$up_df$Method)) {
  a <- adni_plus_list$up_df %>% filter(Method == method)
  b <- adni_list$up_df %>% filter(Method == method)
  
  t_u <- t.test(a$util_score, b$util_score, alternative = "two.sided", var.equal=F)
  t_p <- t.test(a$priv_score, b$priv_score, alternative = "two.sided", var.equal=F)
  res <- rbind(res, data.frame(Method = method,
                               #t_stat_u = t_u$statistic,
                               p_value_u = round(t_u$p.value * nrow(a),3),
                               mean_u_x = t_u$estimate[1],
                               mean_u_y = t_u$estimate[2],
                               #t_stat_p = t_p$statistic,
                               p_value_p = round(t_p$p.value * nrow(a),3),
                               mean_p_x = t_p$estimate[1],
                               mean_p_y = t_p$estimate[2]
                               ))
}


