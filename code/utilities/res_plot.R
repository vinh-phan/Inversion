# Function - plot specification curves 

spec_plot <- function(df, term = "main", estimate = "", threshold = 10, f_stat_filter = -999, aes_setting, color_setting, label_cols, relative_heights = c(0.2, 0.8), 
                      plot_title, note, export_type, file_name) {
  
  if ("f_stat" %in% names(df)) {
    df <- df %>% filter(f_stat > 10)
  }
  # Create plot data frame
  if (term == "main") {
      df_plot <- df %>% 
      filter(!str_detect(term, ":")) %>%
      arrange(estimate) %>% 
      mutate(specification = row_number(), 
             coef_sig = ifelse(p.value < 0.05 & estimate > 0, "sig_pos",
                               ifelse(p.value < 0.05 & estimate < 0, "sig_neg", "insig")), 
             upCI = estimate + 1.96*std.error, 
             loCI = estimate - 1.96*std.error)
  }
  
  else if (term == "interaction") {
    df_plot <- df %>% 
      filter(str_detect(term, ":")) %>%
      arrange(estimate) %>% 
      mutate(specification = row_number(), 
             coef_sig = ifelse(p.value < 0.05 & estimate > 0, "sig_pos",
                               ifelse(p.value < 0.05 & estimate < 0, "sig_neg", "insig")), 
             upCI = estimate + 1.96*std.error, 
             loCI = estimate - 1.96*std.error)
  }
  
  else if (term == "non_coef") {
    df$estimate <- df[[estimate]] 
    df_plot <- df %>% 
      arrange(estimate) %>% 
      mutate(specification = row_number(), 
             coef_sig = ifelse(estimate < threshold, "no", "yes"))
  }
  
  # Plot top panels
  top = df_plot %>% 
    ggplot(aes(specification, estimate, color = coef_sig)) +
    #geom_pointrange(aes(ymin = loCI, ymax = upCI), size = 1, shape = "", alpha = 0.2) +
    geom_point(size = 1) +
    scale_color_manual(values = color_setting) +
    labs(x = "", y = "Estimates") + 
    aes_setting 
  
  # Plot bottom panel 
  bottom = df_plot %>% 
    pivot_longer(
      cols = label_cols, 
      names_to = "variable", 
      values_to = "value"
    ) %>% 
    ggplot(aes(x = specification, 
               y = value, 
               color = coef_sig)) +
    geom_point(aes(x = specification, 
                   y = value), 
               shape = 16, 
               size = 1) + 
    facet_grid(variable ~ 1, scales = "free_y", space = "free_y", 
               labeller = as_labeller(c(
                 fixed_effects_lbl = "Fixed \n effects", 
                 outcome_lbl = "Outcomes", 
                 inv_hour_lbl = "Inv. \n hour", 
                 inv_op_lbl = "Inv. variable", 
                 spatial_agg_lbl = "Spatial \n agg.", 
                 controls_lbl = "Controls",
                 parental_pgs_lbl = "Parental \n PGS", 
                 pol_var_lbl = "Pollution", 
                 tem_agg = "Time \n span"
               )), switch = "y" ) + 
    scale_color_manual(values = color_setting) + 
    labs(x = "Specification number", y = "") + 
    aes_setting + 
    theme(strip.text.x = element_blank())
  
  # Join panels 
  main_plot <- plot_grid(
    top,
    bottom,
    ncol = 1,
    align = "v",
    axis = "l",
    labels = c("A", "B"),
    rel_heights = relative_heights
  )
  
  # Create title 
  title <- ggplot() +
    theme_void() +
    theme(plot.background = element_rect(fill = "white", color = NA)) +
    annotate("text",
             x = 0.5, y = 0.5, 
             label = plot_title,
             size = 6, fontface = "bold", hjust = 0.5)
  
  # Create notes
  note <- ggplot() +
    theme_void() +
    theme(plot.background = element_rect(fill = "white", color = NA)) +
    annotate("text",
             x = 0, y = 0.5, hjust = 0,
             label = note, 
             size = 3)
  
  # Add title and notes
  full_plot <- plot_grid(
    title, 
    main_plot,
    note,
    ncol = 1,
    rel_heights = c(0.1, 1, 0.1)   # height of note area
  )
  
  # Export file 
  if (export_type == "plot_only") {
    ggsave(paste0(output_path, "/", file_name), main_plot, width = 10, height = 9, dpi = 500)
  }
  else {
    ggsave(paste0(output_path, "/", file_name), full_plot, width = 10, height = 9, dpi = 500)
  } 
  
  #return(full_plot)
}











