seeds <- c("42","69","420","666")
file_names <- c("AT_42.csv","AT_69.csv","AT_420.csv","AT_666.csv","Standard_42.csv","Standard_69.csv","Standard_420.csv","Standard_666.csv")
library(ggplot2)
library(ggExtra)
library(patchwork)
options(repr.plot.width=10, repr.plot.height=7)

base_theme <-
    theme_minimal() +
        theme(legend.position = "bottom",
              legend.direction="horizontal",
              legend.text = element_text(size = 12),  # Adjust legend text size
              legend.key.width = unit(0.75, "cm"),  # Adjust width of legend keys (color boxes)
              legend.key.height = unit(0.5, "cm"),
              axis.text = element_text(size = 14),
              axis.title = element_text(size = 16),
              plot.title = element_text(size = 18, face = "bold"),
              strip.text = element_text(size = 16, vjust =2), # Left-align facet labels
              axis.text.y = element_text(size = 14, hjust = 0), # Right-align y-axis labels
              panel.grid.minor = element_blank(),
              # panel.grid.major = element_blank(),
              strip.placement = "outside",
              # Keep the panel border
              # panel.border = element_rect(color = "gray70", fill = NA, size=0.5),
              axis.ticks = element_line()) # Keep tick marks

for(seed in seeds){
    data_rob <- read.csv(paste0("results/AT_",seed,".csv"))%>% mutate(class = as.factor(class))
    data_non_rob <- read.csv(paste0("results/Standard_",seed,".csv"))%>% mutate(class = as.factor(class))
    png(paste0(seed,".svg"), width = 1000, height = 800)
    p_rob<-ggplot(data_rob, aes(x = confidence, y = PGD_robustness, color = class)) +
        geom_point() +
        labs(title = paste("Robust Network ", seed),
             x = "confidence score (original)",
             y = "Adversarial robustness (PGD Steps)") + ylim(c(0,250)) + base_theme
    p_rob_scaled<-ggplot(data_rob, aes(x = scaled_confidence, y = PGD_robustness, color = class)) +
        geom_point() +
        labs(title = paste("Robust Network, temp scaled ", seed),
             x = "confidence score (temperature scaled)",
             y = "Adversarial robustness (PGD Steps)") + ylim(c(0,250)) + base_theme

    p_non_rob<-ggplot(data_non_rob, aes(x = confidence, y = PGD_robustness, color = class)) +
        geom_point() +
        labs(title = paste("Standard Network ", seed),
             x = "confidence score (original)",
             y = "Adversarial robustness (PGD Steps)") + ylim(c(0,250)) + base_theme
    p_non_rob_scaled<-ggplot(data_non_rob, aes(x = scaled_confidence, y = PGD_robustness, color = class)) +
        geom_point() +
        labs(title = paste("Standard Network, temp scaled ", seed),
             x = "confidence score (temperature scaled)",
             y = "Adversarial robustness (PGD Steps)") + ylim(c(0,250)) + base_theme


    dp_rob <- ggplot(data_rob, aes(y = PGD_robustness, fill = class)) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none") + ylim(c(0,250))

    dp_non_rob <- ggplot(data_non_rob, aes(y = PGD_robustness, fill = class)) +
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
