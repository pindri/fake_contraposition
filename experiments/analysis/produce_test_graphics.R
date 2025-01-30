
result_folder <- "/home/peter/tu/papers/verification/fake_contraposition/results/cifar/"
sample_files<-list.files(path = result_folder, pattern = "sampling_c.*0\\..*3125_last.*csv")

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

find_test_file <- function(folder,filename){
    shortname <- substr(filename,10+nchar(folder),1000)
    paste0(folder,list.files(path=folder,pattern=paste0(".*test_",shortname))[1])
}

find_validation_file <- function(folder,filename){
    shortname <- substr(filename,10+nchar(folder),1000)
    paste0(folder,list.files(path=folder,pattern=paste0(".*validation_set_",shortname))[1])
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

rob_preprocessing <- function(tibble){
    tibble %>%
        mutate(#class = as.factor(class),
               pgd_robustness_steps = pgd_robustness_steps,
               pgd_robustness_distances = round(pgd_robustness_distances,digits = 5)
               ) %>%
        arrange(desc(confidence)) %>%
        mutate(min_robustness_steps = cummin(pgd_robustness_steps),
               min_robustness_distances = cummin(pgd_robustness_distances)) %>%
        ungroup()
        # arrange(class)
}

extract_name <- function(filename) {
  filename <- gsub("\\.csv$", "", filename)
  parts <- unlist(strsplit(filename, "_"))
  extracted <- parts[5:7]
  return(extracted)
}
result_table <- list()
# plot_scheme(highcontrast(3), colours = TRUE, names = TRUE, size = 0.9)

i <- 0
for (file in paste0(result_folder,sample_files)){

    # print(nrow(sampling_data))
    print(file)
    sampling_data <- read.csv(file) %>% rob_preprocessing()
    test_data <- find_test_file(result_folder, file) %>% read.csv() %>% rob_preprocessing()
    valid_data <- find_validation_file(result_folder, file) %>% read.csv() %>% rob_preprocessing()
    # svg(paste0(seed,"test.svg"), width=13/3*2,height=10/3*2)
    max_conf <- (sampling_data %>% arrange(confidence) %>%
        .[chernoff_bound_index(nrow(sampling_data),0.01,0.005),])$confidence
    index <- chernoff_bound_index(nrow(sampling_data),0.01,0.005)

    # densityplot_rob <- ggplot(test_data,
    #                           aes(y = pgd_robustness_distances, fill = highcontrast(3)[1], color = highcontrast(3)[1])) +
    #     geom_density(alpha = 0.3) +
    #     theme_void() +
    #     theme(legend.position = "none")

    sampling_ecdf <- ecdf(sampling_data$confidence)
    densityplot_rob <- ggplot(test_data, aes(x = sampling_ecdf(confidence))) +
        geom_density(alpha = 0.3, fill = highcontrast(3)[1],color = highcontrast(3)[1]) + theme_void() +
        theme(legend.position = "none")

    opt <- extract_name(file)[1]
    opt <- ifelse(opt!="AT",paste(opt,"Training"),paste0("TRADES Training"))


    scatterplot_rob <- ggplot() +
        geom_line(data = sampling_data %>%
        mutate(min_robustness_steps =
                   ifelse(rank(confidence) < index, min_robustness_steps,NA),
               min_robustness_distances =
                   ifelse(rank(confidence) < index, min_robustness_distances,NA)),
              aes(x=sampling_ecdf(confidence),y=min_robustness_distances, color = "Lower Bound"),
              linewidth=2) +
        geom_point(data =test_data, aes(x = sampling_ecdf(confidence), y = pgd_robustness_distances, color = "Test Data"), alpha =.2) +
        labs(title = paste("CIFAR10 Lower Bound vs Test Data with",opt),
             x = "Confidence Score (mapped to ECDF)",
             y = "Adversarial Robustness (L inf distance to PGD Example)") +
        ylim(c(0, 0.25)) +
        base_theme +
    geom_vline(aes(xintercept = sampling_ecdf(max_conf), color = "Max Confidence"),  linetype=2, linewidth=1.5)+
    scale_color_manual(values = c("Lower Bound" = highcontrast(3)[3],
                                    "Test Data" = highcontrast(3)[1],
                                    "Max Confidence" = highcontrast(3)[2]),
                       name = "Legend",
                       breaks = c("Lower Bound", "Test Data", "Max Confidence"),
                       labels = c("Lower Bound", "Test Data", "Max Confidence")) +
    theme(legend.position = "bottom") # Adjust legend position as needed

    combo_plot <- list(densityplot_rob,scatterplot_rob) |>
        wrap_plots(nrow = 2,heights=c(1,5))
    ggsave(filename=paste0("plots/cifar/please",file,".svg"),
           plot=scatterplot_rob,
           width=13/3*2,
           height=10/3*2)

    mapping <- sampling_data %>%
        group_by(min_robustness_distances) %>%
        arrange(min_robustness_distances,desc(confidence)) %>%
        filter(row_number()==1) %>% #pick the lowest robustness value and among these the highest confidence
        filter(confidence < max_conf) # this line is tricky, not doing this will allow the consideration of too small confidence ps

    mapping[mapping$confidence>max_conf,]$pgd_robustness_distances <-
        max(mapping[mapping$confidence<max_conf,]$pgd_robustness_distances)

    if(all(mapping$confidence>max_conf)){
        #thank you,
        next
    }

    valid_test_preds <- test_data %>% filter(confidence < max_conf) %>%
        mutate(distance_lower_bound =
                   mapping$pgd_robustness_distances[findInterval(confidence,mapping$confidence)+1],
               violates_bound = pgd_robustness_distances < distance_lower_bound
               )

     valid_sampling_preds <- sampling_data %>% filter(confidence < max_conf) %>%
        mutate(distance_lower_bound =
                   mapping$pgd_robustness_distances[findInterval(confidence,mapping$confidence)+1],
               violates_bound = pgd_robustness_distances < distance_lower_bound
               )

    violating_test_preds <- valid_test_preds %>% filter(violates_bound)
    counter_mass_data <- violating_test_preds %>%
        rowwise() %>%
        mutate(conf = confidence, rob = distance_lower_bound,
               counter_examples = test_data %>%
            filter(test_data$confidence >= max(conf,-1,na.rm=T), test_data$pgd_robustness_distances < min(rob,100,na.rm=T)) %>%
            nrow(), #we count the number of mor confident samples samples with robustness lower this
               mass = test_data %>% filter(test_data$confidence >= max(conf,-1,na.rm=T)) %>% nrow())
    counter_mass_data %>% reframe(counter_examples/mass) %>% max(0)



    i <- i+1
    result_table[[i]]<- c(extract_name(file),nrow(mapping), nrow(valid_test_preds), sum(valid_test_preds$violates_bound ,na.rm=T),
      counter_mass_data %>% reframe(counter_examples/mass) %>% max() %>% max(0))
    # dev.off()
}
results_df <- do.call(rbind, result_table)

# Convert to data frame with meaningful column names
colnames(results_df) <- c("opt_func", "seed", "rob_beta", "map_size", "valid_preds", "total_counterexamples", "relative_error")
results_df <- as.data.frame(results_df)

write.csv(results_df, "summary_cifar_pgd_please.csv", row.names = FALSE)
#======= functions to check the estimated mass of counterexamples per sample guarantee =====
#
# result_data_preprocessing <- function(x){
#         arrange(x,desc(confidence)) %>%
#         mutate(min_robustness = cummin(PGD_robustness)) %>%
#         arrange(desc(confidence)) %>%
#         mutate(min_scaled_robustness = cummin(PGD_robustness))
# }
#
#
# counter_mass <- function(N,conf,rob){
#     return (N%>% filter(scaled_confidence >= conf, PGD_robustness <rob) %>% nrow())
# }
#
# right_mass <- function(N,conf){
#     return (N%>% filter(scaled_confidence >= conf) %>% nrow())
# }
#
# lookup_guarantee <- function(N, new_confidence) {
#     N <- N %>% filter(PGD_robustness == min_scaled_robustness)
#     N_sorted <- N %>% arrange(scaled_confidence,PGD_robustness)
#     sorted_confidences <- N_sorted$scaled_confidence
#     indices <- findInterval(new_confidence, sorted_confidences, rightmost.closed = FALSE) + 1
#     indices[indices > nrow(N_sorted)] <- NA
#     N_sorted$min_scaled_robustness[indices]
# }
# seeds <- c("666","42","69")
# for(seed in seeds){
#     sample<- read.csv(paste0("results_cifar/Standard_",seed,".csv"))%>%
#         result_data_preprocessing()
#     test <- read.csv(paste0("results_cifar/results/Standard_",seed,".csv"))%>%
#         result_data_preprocessing()
#     max_conf <- (sample %>% arrange(scaled_confidence) %>%
#         .[chernoff_bound_index(nrow(sample),0.01,0.01),])$scaled_confidence
#     # get guarantees
#     test2 <- test %>%
#     arrange(scaled_confidence) %>%
#     mutate(lower_bound = lookup_guarantee(sample,scaled_confidence))
#     print(paste("scaled confidence",seed))
#     if(test2 %>% filter(PGD_robustness < lower_bound, scaled_confidence < max_conf) %>% nrow() ==0){
#         print("no counterexamples")
#     }else{
#         test3<- test2 %>% filter(PGD_robustness < lower_bound, scaled_confidence < max_conf) %>%
#         rowwise() %>%
#         mutate(counter_examples = counter_mass(test,scaled_confidence,lower_bound),
#                more_confident = right_mass(test,scaled_confidence)) %>%
#         summarize(relative_error = counter_examples/more_confident)
#         print(test3$relative_error)
#         print(max(test3$relative_error))
#
#     }
#     sample$scaled_confidence <- sample$confidence
#     sample$min_scaled_robustness <- sample$min_robustness
# # <- mutate(scaled_confidence = confidence, min_scaled_robustness = min_robustness)
#     test$scaled_confidence <- test$confidence
#     test$min_scaled_robustness <- test$min_robustness
#     max_conf <- (sample %>% arrange(scaled_confidence) %>%
#         .[chernoff_bound_index(nrow(sample),0.01,0.01),])$scaled_confidence
#     test2 <- test %>%
#     arrange(scaled_confidence) %>%
#     mutate(lower_bound = lookup_guarantee(sample,scaled_confidence))
#
#     print(paste("unscaled confidence",seed))
#     if(test2 %>% filter(PGD_robustness < lower_bound, scaled_confidence < max_conf) %>% nrow() ==0){
#         print("no counterexamples")
#         next
#     }
#     test3<- test2 %>% filter(PGD_robustness < lower_bound, scaled_confidence <= max_conf) %>%
#     rowwise() %>%
#     mutate(counter_examples = counter_mass(test,scaled_confidence,lower_bound),
#            more_confident = right_mass(test,scaled_confidence)) %>%
#     summarize(relative_error = counter_examples/more_confident)
#     print(test3$relative_error)
#     print(max(test3$relative_error))
#
# }