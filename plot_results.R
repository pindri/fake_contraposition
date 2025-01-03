seeds <- c("42","69","420","666")
file_names <- c("AT_42.csv","AT_69.csv","AT_420.csv","AT_666.csv","Standard_42.csv","Standard_69.csv","Standard_420.csv","Standard_666.csv")
library(ggplot2)
library(ggExtra)
library(patchwork)
library(dplyr)
library(khroma)
options(repr.plot.width=10, repr.plot.height=7)

base_theme <-
    theme_minimal() +
        theme(legend.position = "bottom",
              legend.direction="horizontal",
              legend.text = element_text(size = 16),  # Adjust legend text size
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
highcontrast <- color("high contrast")
# plot_scheme(highcontrast(3), colours = TRUE, names = TRUE, size = 0.9)
for(seed in seeds){
    data_rob<- read.csv(paste0("results/AT_",seed,".csv"))%>%
        mutate(Class = as.factor(class)) %>%
        mutate(Class = ifelse(Class %in% c(0,7), Class,"Rest")) %>%
        group_by(as.factor(Class)) %>%
        arrange(desc(confidence)) %>%
        mutate(min_robustness = cummin(PGD_robustness)) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(min_scaled_robustness = cummin(PGD_robustness))

    data_rob_test <- read.csv(paste0("results/test_AT_",seed,".csv"))%>%
        arrange(desc(scaled_confidence)) %>%
        mutate(Class = as.factor(class)) %>%
        mutate(Class = ifelse(Class %in% c(0,7), Class,"Rest")) %>%
        group_by(as.factor(Class)) %>%
        arrange(desc(confidence)) %>%
        mutate(min_robustness = cummin(PGD_robustness)) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(min_scaled_robustness = cummin(PGD_robustness))

        # head(n=2000) %>%
    data_non_rob <- read.csv(paste0("results/Standard_",seed,".csv")) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(Class = as.factor(class)) %>%
        mutate(Class = ifelse(Class %in% c(0,7), Class,"Rest")) %>%
        group_by(as.factor(Class)) %>%
        arrange(desc(confidence)) %>%
        mutate(min_robustness = cummin(PGD_robustness)) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(min_scaled_robustness = cummin(PGD_robustness))
    svg(paste0(seed,".svg"), width=13,height=10)
    p_rob<-ggplot(data_rob%>% sample_frac(0.005), aes(x = confidence, y = PGD_robustness, color = Class)) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_point(alpha =0.2) +
        labs(title = paste("Robust Training, no Scaling "),
             x = "Confidence Score (Original)",
             y = "Adversarial Robustness (PGD Steps)") + ylim(c(0,200)) + base_theme +
        guides(color = guide_legend(override.aes = list(alpha = 1)))
        # geom_line(data = data_rob,aes(x=confidence,y=min_robustness,color=Class), linewidth=2)
    p_rob_scaled<-ggplot(data_rob %>% sample_frac(0.005), aes(x = scaled_confidence, y = PGD_robustness, color = Class)) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_point(alpha =0.2) +
        labs(title = paste("Robust Training, Temperature Scaling"),
             x = "Confidence Score (Temperature Scaled)",
             y = "Adversarial Robustness (PGD Steps)") + ylim(c(0,200)) + base_theme +
        guides(color = guide_legend(override.aes = list(alpha = 1))) +
                geom_line(data = data_rob,aes(x=scaled_confidence,y=min_scaled_robustness,color=Class), linewidth=2)

    p_non_rob<-ggplot(data_non_rob %>% sample_frac(0.005), aes(x = confidence, y = PGD_robustness, color = Class)) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_point(alpha =0.2) +
        labs(title = paste("Standard Training, no Scaling "),
             x = "Confidence Score (Original)",
             y = "Adversarial Robustness (PGD Steps)") + ylim(c(0,200)) +guides(color = guide_legend(override.aes = list(alpha = 1))) +  base_theme
                # geom_line(data = data_non_rob,aes(x=confidence,y=min_robustness,color=Class), linewidth=2)

    p_non_rob_scaled<-ggplot(data_non_rob %>% sample_frac(0.005), aes(x = scaled_confidence, y = PGD_robustness, color = Class)) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_point(alpha =0.2) +
        labs(title = paste("Standard Training, Temperature Scaling"),
             x = "Confidence Score (Temperature Scaled)",
             y = "Adversarial Robustness (PGD Steps)") + ylim(c(0,200)) + guides(color = guide_legend(override.aes = list(alpha = 1))) + base_theme
                # geom_line(data = data_non_rob,aes(x=scaled_confidence,y=min_scaled_robustness,color=Class), linewidth=2)


    dp_rob <- ggplot(data_rob, aes(y = PGD_robustness, fill = Class)) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none") + ylim(c(0,200))

    dp_non_rob <- ggplot(data_non_rob, aes(y = PGD_robustness, fill = Class)) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none") + ylim(c(0,200))

    dp_rob_conf <- ggplot(data_rob, aes(x = confidence, fill = as.factor(Class))) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_rob_conf_scaled <- ggplot(data_rob, aes(x = scaled_confidence, fill = as.factor(Class))) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_non_rob_conf <- ggplot(data_non_rob, aes(x = confidence, fill = as.factor(Class))) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_non_rob_conf_scaled <- ggplot(data_non_rob, aes(x = scaled_confidence, fill = as.factor(Class))) +
        scale_fill_manual(values = rev(highcontrast(3))) +
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
