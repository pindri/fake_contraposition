seeds <- c("42","69","666")
file_names <- c("AT_42.csv","AT_69.csv","AT_420.csv","AT_666.csv","Standard_42.csv","Standard_69.csv","Standard_420.csv","Standard_666.csv")
library(ggplot2)
library(ggExtra)
library(patchwork)
library(dplyr)
library(khroma)
options(repr.plot.width=10, repr.plot.height=7)

chernoff_bound_index <-  function(n,p,delta){
    ceiling(n*(1-p)-sqrt(2*n*(1-p)*log(2/delta)))
}

result_data_preprocessing <- function(x){
    mutate(x,Class = as.factor(class)) %>%
        mutate(Class = ifelse(Class %in% c(0,7), Class,"Rest")) %>%
        group_by(as.factor(Class)) %>%
        arrange(x,desc(confidence)) %>%
        mutate(min_robustness = cummin(PGD_robustness)) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(min_scaled_robustness = cummin(PGD_robustness))
}

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
        mutate(min_scaled_robustness = cummin(PGD_robustness)) %>%
        ungroup() %>% arrange(Class)

    data_rob_test <- read.csv(paste0("results/test_AT_",seed,".csv"))%>%
        arrange(desc(scaled_confidence)) %>%
        mutate(Class = as.factor(class)) %>%
        mutate(Class = ifelse(Class %in% c(0,7), Class,"Rest")) %>%
        group_by(as.factor(Class)) %>%
        arrange(desc(confidence)) %>%
        mutate(min_robustness = cummin(PGD_robustness)) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(min_scaled_robustness = cummin(PGD_robustness)) %>%
        ungroup() %>% arrange(Class)

        # head(n=500) %>%
    data_non_rob <- read.csv(paste0("results/Standard_",seed,".csv")) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(Class = as.factor(class)) %>%
        mutate(Class = ifelse(Class %in% c(0,7), Class,"Rest")) %>%
        group_by(as.factor(Class)) %>%
        arrange(desc(confidence)) %>%
        mutate(min_robustness = cummin(PGD_robustness)) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(min_scaled_robustness = cummin(PGD_robustness))%>%
        ungroup() %>% arrange(Class)
    svg(paste0(seed,".svg"), width=13,height=10)
    p_rob<-ggplot(data_rob%>% sample_frac(0.005), aes(x = confidence, y = PGD_robustness, color = Class)) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_point(alpha =0.2) +
        labs(title = paste("Robust Training, no Scaling "),
             x = "Confidence Score (Original)",
             y = "Adversarial Robustness (PGD Steps)") + ylim(c(0,200)) + base_theme +
        guides(color = guide_legend(override.aes = list(alpha = 1)))#+
                # geom_line(data = data_rob,aes(x=scaled_confidence,y=min_scaled_robustness,color=Class), linewidth=2)

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


    dp_rob <- ggplot(data_rob, aes(y = PGD_robustness, fill = Class, color = Class)) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none") + ylim(c(0,200))

    dp_non_rob <- ggplot(data_non_rob, aes(y = PGD_robustness, fill = Class, color = Class)) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none") + ylim(c(0,200))

    dp_rob_conf <- ggplot(data_rob, aes(x = confidence, fill = as.factor(Class), color = Class)) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_rob_conf_scaled <- ggplot(data_rob, aes(x = scaled_confidence, fill = as.factor(Class), color = Class)) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_non_rob_conf <- ggplot(data_non_rob, aes(x = confidence, fill = as.factor(Class), color = Class)) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        scale_color_manual(values = rev(highcontrast(3))) +
        geom_density(alpha = 0.3) + theme_void() +
        theme(legend.position = "none")

    dp_non_rob_conf_scaled <- ggplot(data_non_rob, aes(x = scaled_confidence, fill = as.factor(Class), color = Class)) +
        scale_fill_manual(values = rev(highcontrast(3))) +
        scale_color_manual(values = rev(highcontrast(3))) +
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

svg(paste0(seed,"test.svg"), width=13/3*2,height=10/3*2)
max_conf <- (data_rob %>% arrange(scaled_confidence) %>% .[chernoff_bound_index(670000,0.01,0.01),])$scaled_confidence
max_rob <- min((data_rob %>% filter(scaled_confidence>max_conf))$min_scaled_robustness)

dp_rob_conf_scaled <- ggplot(data_rob_test, aes(x = scaled_confidence)) +
        geom_density(alpha = 0.3, fill = highcontrast(3)[1],color = highcontrast(3)[1]) + theme_void() +
        theme(legend.position = "none")
redp <- ggplot() +
    geom_line(data = data_rob %>% filter(Class == "Rest") %>% mutate(min_scaled_robustness = ifelse(min_scaled_robustness < max_rob,min_scaled_robustness,max_rob)),
              aes(x=scaled_confidence,y=min_scaled_robustness, color = "Lower Bound"),
              linewidth=2) +
        geom_point(data =data_rob_test, aes(x = scaled_confidence, y = PGD_robustness, color = "Test Data"),alpha =.2) +
        labs(title = paste("Lower Bound on Test Data"),
             x = "Confidence Score (Scaled)",
             y = "Adversarial Robustness (PGD Steps)") + ylim(c(0,200)) + base_theme +
    geom_vline(aes(xintercept = max_conf, color = "Max Confidence"),  linetype=2, linewidth=1.5)+
    scale_color_manual(values = c("Lower Bound" = highcontrast(3)[3],
                                    "Test Data" = highcontrast(3)[1],Max
                                    "Max Confidence" = highcontrast(3)[2]),
                       name = "Legend",
                       breaks = c("Lower Bound", "Test Data", "Max Confidence"),
                       labels = c("Lower Bound", "Test Data", "Max Confidence")) +
    theme(legend.position = "bottom") # Adjust legend position as needed
    list(dp_rob_conf_scaled,redp) |>
  wrap_plots(nrow = 2,heights=c(1,5))
    dev.off()





#======= functions to check the estimated mass of counterexamples per sample guarantee =====

result_data_preprocessing <- function(x){
        arrange(x,desc(confidence)) %>%
        mutate(min_robustness = cummin(PGD_robustness)) %>%
        arrange(desc(scaled_confidence)) %>%
        mutate(min_scaled_robustness = cummin(PGD_robustness))
}


counter_mass <- function(N,conf,rob){
    return (N%>% filter(scaled_confidence >= conf, PGD_robustness <rob) %>% nrow())
}

right_mass <- function(N,conf){
    return (N%>% filter(scaled_confidence >= conf) %>% nrow())
}

lookup_guarantee <- function(N, new_confidence) {
    N <- N %>% filter(PGD_robustness == min_scaled_robustness)
    N_sorted <- N %>% arrange(scaled_confidence,PGD_robustness)
    sorted_confidences <- N_sorted$scaled_confidence
    indices <- findInterval(new_confidence, sorted_confidences, rightmost.closed = FALSE) + 1
    indices[indices > nrow(N_sorted)] <- NA
    N_sorted$min_scaled_robustness[indices]
}
seeds <- c("666","42","69")
for(seed in seeds){
    sample<- read.csv(paste0("results_cifar/Standard_",seed,".csv"))%>%
        result_data_preprocessing()
    test <- read.csv(paste0("results_cifar/results/Standard_",seed,".csv"))%>%
        result_data_preprocessing()
    max_conf <- (sample %>% arrange(scaled_confidence) %>%
        .[chernoff_bound_index(nrow(sample),0.01,0.01),])$scaled_confidence
    # get guarantees
    test2 <- test %>%
    arrange(scaled_confidence) %>%
    mutate(lower_bound = lookup_guarantee(sample,scaled_confidence))
    print(paste("scaled confidence",seed))
    if(test2 %>% filter(PGD_robustness < lower_bound, scaled_confidence < max_conf) %>% nrow() ==0){
        print("no counterexamples")
    }else{
        test3<- test2 %>% filter(PGD_robustness < lower_bound, scaled_confidence < max_conf) %>%
        rowwise() %>%
        mutate(counter_examples = counter_mass(test,scaled_confidence,lower_bound),
               more_confident = right_mass(test,scaled_confidence)) %>%
        summarize(relative_error = counter_examples/more_confident)
        print(test3$relative_error)
        print(max(test3$relative_error))

    }
    sample$scaled_confidence <- sample$confidence
    sample$min_scaled_robustness <- sample$min_robustness
# <- mutate(scaled_confidence = confidence, min_scaled_robustness = min_robustness)
    test$scaled_confidence <- test$confidence
    test$min_scaled_robustness <- test$min_robustness
    max_conf <- (sample %>% arrange(scaled_confidence) %>%
        .[chernoff_bound_index(nrow(sample),0.01,0.01),])$scaled_confidence
    test2 <- test %>%
    arrange(scaled_confidence) %>%
    mutate(lower_bound = lookup_guarantee(sample,scaled_confidence))

    print(paste("unscaled confidence",seed))
    if(test2 %>% filter(PGD_robustness < lower_bound, scaled_confidence < max_conf) %>% nrow() ==0){
        print("no counterexamples")
        next
    }
    test3<- test2 %>% filter(PGD_robustness < lower_bound, scaled_confidence <= max_conf) %>%
    rowwise() %>%
    mutate(counter_examples = counter_mass(test,scaled_confidence,lower_bound),
           more_confident = right_mass(test,scaled_confidence)) %>%
    summarize(relative_error = counter_examples/more_confident)
    print(test3$relative_error)
    print(max(test3$relative_error))

}