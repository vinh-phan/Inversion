# Functions - Run all regressions based on specification dataframe 

############################################
### 2SLS regressions 
run_2sls_specs <- function(df, ss_out_file, fs_out_file) {
  df_split <- split(df, df$hour)
  
  # open connection for writing (overwrite file if exists)
  first_write_ss <- TRUE
  first_write_fs <- TRUE
  
  for (i in seq_len(nrow(df_filtered_specs))) {
    message("Estimating 2SLS spec ", i, " of ", nrow(df_filtered_specs))
    
    # Extracting specification
    spec <- df_filtered_specs[i, ]
    
    # Extracting the data set with the right inversion hour 
    df_use <- df_split[[ spec$inv_hour ]]
    
    # Estimating the regression 
    mod <- feols(as.formula(spec$formula), data = df_use)
    
    # ====== Extracting the 2SLS estimates ====
    ss_est <- tidy(mod) %>%
      filter(str_detect(term, "pm")) %>%
      mutate(spec_id = spec$spec_id, 
             stage = "second")
    
    # Extracting the F-stat 
    fstat <- fitstat(mod, "ivf")[[1]][["stat"]]
    ss_est$f_stat <- fstat
    
    # Merge results with spec metadata
    ss_est <- left_join(ss_est, spec, by = c("spec_id")) %>%
      mutate(across(where(is.list), ~ sapply(., paste, collapse = ", ")))
    
    # ====== Extracting the 1st-stage estimates ====
    fs_est <- purrr::imap_dfr(
      mod$iv_first_stage,
      ~ tidy(.x) %>%
        mutate(
          endogenous = .y,
          spec_id = spec$spec_id,
          stage = "first"
        )
    ) %>%
      filter(
        # Only keeping relevant terms
        ( # Main effects 
          str_detect(term, "t_dif") &
            str_detect(endogenous, "pm") &
            !str_detect(term, ":") &
            !str_detect(endogenous, ":")
        )
        |
          # Interaction
          (
            str_detect(term, ":") &
              str_detect(endogenous, ":")
          )
      ) %>% 
      filter(
        !str_detect(endogenous, "(maternal|paternal)")
      ) %>% 
      filter(
        !str_detect(term, "(maternal|paternal)")
      ) %>%
      left_join(spec, by = "spec_id")
    
    # ==== Export datasets ====
    # 2nd stage
    if (nrow(ss_est) > 0) {
      if (first_write_ss) {
        write_parquet(ss_est, ss_out_file)
        first_write_ss <- FALSE
      } else {
        # append mode: collect old + new, then rewrite
        ss_old <- read_parquet(ss_out_file)
        write_parquet(bind_rows(ss_old, ss_est), ss_out_file)
      }
    }
    
    # 1st stage
    if (nrow(fs_est) > 0) {
      if (first_write_fs) {
        write_parquet(fs_est, fs_out_file)
        first_write_fs <- FALSE
      } else {
        # append mode: collect old + new, then rewrite
        fs_old <- read_parquet(fs_out_file)
        write_parquet(bind_rows(fs_old, fs_est), fs_out_file)
      }
    }
  }
}

############################################
### Checking for left-over variation 
run_var_specs <- function(df, out_file) {
  df_split <- split(df, df$hour)
  
  # open connection for writing (overwrite file if exists)
  first_write <- TRUE
  
  for (i in seq_len(nrow(df_filtered_specs))) {
    message("Checking variation: spec ", i, " of ", nrow(df_filtered_specs))
    
    spec <- df_filtered_specs[i, ]
    df_use <- df_split[[ spec$inv_hour ]]
    
    # Regress Instrument on Weather Controls and FE 
    formula <- ifelse(spec$controls == "", 
                          paste0(spec$inv_op, " ~  1 | ", spec$fixed_effects), 
                          paste0(spec$inv_op, " ~ ", spec$controls, " | ", spec$fixed_effects))
    
    mod <- feols(as.formula(formula), data = df_use)
    
    df_use$Zt <- NA
    df_use$Zt[obs(mod)] <- resid(mod)
    var_Zt <- var(df_use$Zt, na.rm = TRUE) # Variance of D_t 
    r2_Zt = r2(mod, "r2") # R-squared 

    # Regress Outcome on Weather controls and FE 
    formula <- ifelse(spec$controls == "", 
                          paste0(spec$outcome, " ~  1 | ", spec$fixed_effects), 
                          paste0(spec$outcome, " ~ ", spec$controls, " | ", spec$fixed_effects))
    
    mod <- feols(as.formula(formula), data = df_use)
    
    df_use$Yt <- NA
    df_use$Yt[obs(mod)] <- resid(mod)
    var_Yt <- var(df_use$Yt, na.rm = TRUE) # Variance of Z_t
    r2_Yt = r2(mod, "r2") # R-squared 
    
    # Regress Y tilde on Z tilde 
    mod <- feols(as.formula("Yt ~ Zt"), data = df_use)
    
    var_Ythat <- var(predict(mod), na.rm = TRUE) # Variance of D tilde hat 
    r2_partial = r2(mod, "r2") # Partial R-squared
    
    # merge results with spec metadata
    spec_id <- spec$spec_id 
    df_variation <- as.data.frame(cbind(var_Yt, r2_Yt, var_Zt, r2_Zt, var_Ythat, r2_partial, spec_id))
    
    # append to parquet incrementally
    if (nrow(df_variation) > 0) {
      if (first_write) {
        write_parquet(df_variation, out_file)
        first_write <- FALSE
      } else {
        # append mode: collect old + new, then rewrite
        old <- read_parquet(out_file)
        write_parquet(bind_rows(old, df_variation), out_file)
      }
    }
  }
}


############################################
### General regressions 
run_specs <- function(df, out_file, coef_term = "t_dif") {
  
  # open connection for writing (overwrite file if exists)
  first_write <- TRUE
  
  for (i in seq_len(nrow(df_filtered_specs))) {
    message("Estimating association spec ", i, " of ", nrow(df_filtered_specs))
    
    spec <- df_filtered_specs[i, ]
    
    if (coef_term == "t_dif") {
      df_split <- split(df, df$hour)
      df_use <- df_split[[ spec$inv_hour ]]
      mod <- feols(as.formula(spec$formula), data = df_use)
      
      est <- tidy(mod) %>%
        filter(str_detect(term, coef_term)) %>%
        mutate(spec_id = spec$spec_id)
    }
    else if (coef_term == "pm") {
      df_use <- df %>% filter(hour == "hour_2am") # Arbitrary hour to get unique obs. 
      mod <- feols(as.formula(spec$formula), data = df_use)
      
      est <- tidy(mod) %>%
        filter(str_detect(term, coef_term)) %>%
        mutate(spec_id = spec$spec_id)
    }
    
    est$ar2_Y_ass = r2(mod, "r2") # Adjusted R-squared

    # merge results with spec metadata
    out <- left_join(est, spec, by = c("spec_id")) %>%
      mutate(across(where(is.list), ~ sapply(., paste, collapse = ", ")))
    
    # append to parquet incrementally
    if (nrow(out) > 0) {
      if (first_write) {
        write_parquet(out, out_file)
        first_write <- FALSE
      } else {
        # append mode: collect old + new, then rewrite
        old <- read_parquet(out_file)
        write_parquet(bind_rows(old, out), out_file)
      }
    }
  }
}






















