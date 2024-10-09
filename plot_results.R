seeds <- c("42","69","420","666")
file_names <- c("AT_42.csv","AT_69.csv","AT_420.csv","AT_666.csv","Standard_42.csv","Standard_69.csv","Standard_420.csv","Standard_666.csv")
library(ggplot2)
library(ggExtra)
library(patchwork)
options(repr.plot.width=10, repr.plot.height=7)
for(seed in seeds){
    data_rob <- read.csv(paste0("results/AT_",seed,".csv"))
    data_non_rob <- read.csv(paste0("results/Standard_",seed,".csv"))
    png(paste0(seed,".png"), width = 1000, height = 800)
    p_rob<-ggplot(data_rob, aes(x = confidence, y = PGD_robustness, color = as.factor(class))) +
        geom_point() +
        labs(title = paste("Robust Network ", seed),
             x = "confidence score (original)",
             y = "Adversarial robustness (PGD Steps)") + ylim(c(0,250))
    p_rob_scaled<-ggplot(data_rob, aes(x = scaled_confidence, y = PGD_robustness, color = as.factor(class))) +
        geom_point() +
        labs(title = paste("Robust Network, temp scaled ", seed),
             x = "confidence score (temperature scaled)",
             y = "Adversarial robustness (PGD Steps)") + ylim(c(0,250))

    p_non_rob<-ggplot(data_non_rob, aes(x = confidence, y = PGD_robustness, color = as.factor(class))) +
        geom_point() +
        labs(title = paste("Standard Network ", seed),
             x = "confidence score (original)",
             y = "Adversarial robustness (PGD Steps)") + ylim(c(0,250))
    p_non_rob_scaled<-ggplot(data_non_rob, aes(x = scaled_confidence, y = PGD_robustness, color = as.factor(class))) +
        geom_point() +
        labs(title = paste("Standard Network, temp scaled ", seed),
             x = "confidence score (temperature scaled)",
             y = "Adversarial robustness (PGD Steps)") + ylim(c(0,250))


    dp_rob <- ggplot(data_rob, aes(y = PGD_robustness, fill = as.factor(class))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none") + ylim(c(0,250))

    dp_non_rob <- ggplot(data_non_rob, aes(y = PGD_robustness, fill = as.factor(class))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none") + ylim(c(0,250))

    dp_rob_conf <- ggplot(data_rob, aes(x = confidence, fill = as.factor(class))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_rob_conf_scaled <- ggplot(data_rob, aes(x = scaled_confidence, fill = as.factor(class))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_non_rob_conf <- ggplot(data_non_rob, aes(x = confidence, fill = as.factor(class))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_non_rob_conf_scaled <- ggplot(data_non_rob, aes(x = scaled_confidence, fill = as.factor(class))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")


    empty <- ggplot() + theme_void()
    list(dp_non_rob_conf,dp_non_rob_conf_scaled,empty,
         p_non_rob, p_non_rob_scaled, dp_non_rob,
         dp_rob_conf,dp_rob_conf_scaled,empty,
         p_rob, p_rob_scaled, dp_rob) |>
  wrap_plots(nrow = 4, widths = c(5, 5, 1), heights=c(1,5,1,5))
    dev.off()
}
