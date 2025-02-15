# i need to make it so it splits into intervals, then I'm going to add the whole season data to it
# i can only use the 2021 and 22 seasons as I have the future batters faced and then it will be used
# to predict the 2024 season based on the 2023 stats
# data split up in predicting the full seasons works as it doesn't bother with 2023 to predict 2024

# Load necessary libraries
library(dplyr)
library(tidyr)
library(tidyverse)
library(xgboost)
library(caret)
library(ggplot2)
setwd("~/Desktop/Syracuse/Spring 2025/SABR 2/hackathon")
# Load pitcher stats data
df <- read_csv('savant_data_2021_2023.csv')
df$year <- format(df$game_date, "%Y")
# rm(list=setdiff(ls(), "df"))
####################################################################################################
# Function to create 5-game stretches
generate_5_game_stretches <- function(df) {
  df %>%
    group_by(pitcher, year) %>%
    mutate(
      game_order = dense_rank(game_date),  # Order by date
      stretch_id = ceiling(game_order / 9)  # Assign stretch ID for 5-game intervals
    ) %>%
    group_by(pitcher, year, stretch_id) %>%
    summarise(
      games_played = n_distinct(game_date),
      outs_recorded = sum(
        events %in% c(
          "strikeout", "field_out", "force_out", "sac_fly", "sac_bunt",
          "fielders_choice_out", "other_out", "caught_stealing_3b", "caught_stealing_2b",
          "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b", ""), na.rm = TRUE
      ) +
        (2 * sum(events %in% c(
          "double_play", "grounded_into_double_play", "strikeout_double_play",
          "sac_fly_double_play", "sac_bunt_double_play"
        ), na.rm = TRUE)) +
        (3 * sum(events == "triple_play", na.rm = TRUE)),
      innings_pitched = outs_recorded / 3,
      strikeouts = sum(events == "strikeout", na.rm = TRUE),
      walks = sum(events == "walk", na.rm = TRUE),
      hits_allowed = sum(events %in% c("single", "double", "triple", "home_run"), na.rm = TRUE),
      HR = sum(events == "home_run", na.rm = TRUE),
      hit_batters = sum(events == "hit_by_pitch", na.rm = TRUE),
      runs_allowed = sum(ifelse(inning_topbot == "Top", post_away_score - away_score, 
                                post_home_score - home_score), na.rm = TRUE),
      batters_faced = sum(events %in% c(
        "strikeout", "field_out", "walk", "force_out", "sac_fly", "single",
        "hit_by_pitch", "double", "grounded_into_double_play", "home_run",
        "fielders_choice", "field_error", "triple", "strikeout_double_play",
        "fielders_choice_out", "double_play", "catcher_interf", "sac_bunt"
      ), na.rm = TRUE),# Target variable for next stretch 
      avg_ev = mean(launch_speed, na.rm = TRUE),
      whip = (walks + hits_allowed) / ifelse(innings_pitched > 0, innings_pitched, 1),
      era = ifelse(innings_pitched > 0, (runs_allowed / innings_pitched) * 9, 0),
      k_per_9 = ifelse(innings_pitched > 0, (strikeouts / innings_pitched) * 9, 0),
      bb_per_9 = ifelse(innings_pitched > 0, (walks / innings_pitched) * 9, 0),
      hr_per_9 = ifelse(innings_pitched > 0, (HR / innings_pitched) * 9, 0),
      starts_in_stretch = sum(sp_indicator > 0 & !duplicated(game_date))  
    ) %>%
    ungroup()
}


# Create 5-game stretches
df_stretches <- generate_5_game_stretches(df)

# Prepare training data
df_train_1 <- df_stretches %>%
  group_by(pitcher, year) %>%
  mutate(next_batters_faced = lead(batters_faced)) %>%
  filter(!is.na(next_batters_faced)) %>%
  ungroup()

####################################################################################################
# Function to calculate season-level statistics
calculate_season_stats <- function(df) {
  df  %>%
    group_by(pitcher, year) %>%
    summarise(
      stretch_id = 1,
      games_played = n_distinct(game_date),
      outs_recorded = sum(
        events %in% c(
          "strikeout", "field_out", "force_out", "sac_fly", "sac_bunt",
          "fielders_choice_out", "other_out", "caught_stealing_3b", "caught_stealing_2b",
          "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b", ""), na.rm = TRUE
      ) +
        (2 * sum(events %in% c(
          "double_play", "grounded_into_double_play", "strikeout_double_play",
          "sac_fly_double_play", "sac_bunt_double_play"
        ), na.rm = TRUE)) +
        (3 * sum(events == "triple_play", na.rm = TRUE)),
      innings_pitched = outs_recorded / 3,
      strikeouts = sum(events == "strikeout", na.rm = TRUE),
      walks = sum(events == "walk", na.rm = TRUE),
      hits_allowed = sum(events %in% c("single", "double", "triple", "home_run"), na.rm = TRUE),
      HR = sum(events == "home_run", na.rm = TRUE),
      hit_batters = sum(events == "hit_by_pitch", na.rm = TRUE),
      runs_allowed = sum(ifelse(inning_topbot == "Top", post_away_score - away_score, 
                                post_home_score - home_score), na.rm = TRUE),
      batters_faced = sum(events %in% c(
        "strikeout", "field_out", "walk", "force_out", "sac_fly", "single",
        "hit_by_pitch", "double", "grounded_into_double_play", "home_run",
        "fielders_choice", "field_error", "triple", "strikeout_double_play",
        "fielders_choice_out", "double_play", "catcher_interf", "sac_bunt"
      ), na.rm = TRUE),# Target variable for next season
      avg_ev = mean(launch_speed, na.rm = TRUE),
      whip = (walks + hits_allowed) / ifelse(innings_pitched > 0, innings_pitched, 1),
      era = ifelse(innings_pitched > 0, (runs_allowed / innings_pitched) * 9, 0),
      k_per_9 = ifelse(innings_pitched > 0, (strikeouts / innings_pitched) * 9, 0),
      bb_per_9 = ifelse(innings_pitched > 0, (walks / innings_pitched) * 9, 0),
      hr_per_9 = ifelse(innings_pitched > 0, (HR / innings_pitched) * 9, 0),
      starts_in_stretch = sum(sp_indicator > 0 & !duplicated(game_date))  
    ) %>%
    ungroup()
}

# Create season-level dataset
df_season <- calculate_season_stats(df)
# Prepare data for prediction (using previous season to predict next season)
df_train <- df_season %>%
  group_by(pitcher) %>%
  mutate(next_batters_faced = lead(batters_faced)) %>%
  filter(!is.na(next_batters_faced)) %>%
  ungroup()

df_train <- df_train %>%
  mutate(is_season_level = TRUE)  
df_train_1 <- df_train_1 %>%
  mutate(is_season_level = FALSE)
df_train_2 <- rbind(df_train,df_train_1)
df_train_2 <- rbind(df_train, df_train_1) %>%
  mutate(sample_weight = ifelse(is_season_level, 5, 1))  # Give full-season stats higher weight
####################################################################################################

# Split into training and testing sets
set.seed(937848)
train_index <- createDataPartition(df_train_2$next_batters_faced, p = 0.7, list = FALSE)
train_data <- df_train_2[train_index, ]
test_data <- df_train_2[-train_index, ]

# Convert to matrix format for XGBoost
train_matrix <- as.matrix(train_data %>% select(-c(pitcher, year, stretch_id, next_batters_faced)))
test_matrix <- as.matrix(test_data %>% select(-c(pitcher, year, stretch_id, next_batters_faced)))

train_labels <- log(train_data$next_batters_faced)
test_labels <- log(test_data$next_batters_faced)

train_weights <- train_data$sample_weight  # Subset weights for training
# Perform random search for hyperparameter tuning
set.seed(937848)
n_iter <- 300
param_grid <- data.frame(
  max_depth = sample(4:8, n_iter, replace = TRUE),    # Keep trees moderately deep
  eta = runif(n_iter, 0.005, 0.3),                    # Lower learning rate
  nrounds = sample(500:3000, n_iter, replace = TRUE),# More boosting rounds
  min_child_weight = sample(10:30, n_iter, replace = TRUE),  # Stronger regularization
  subsample = runif(n_iter, 0.6, 0.9),                # Slight randomness in training data selection
  colsample_bytree = runif(n_iter, 0.6, 0.9),         # Prevent too much reliance on all features
  gamma = runif(n_iter, 0.5, 5),                      # Discourage unnecessary splits
  lambda = runif(n_iter, 1, 10),                      # Stronger L2 regularization
  alpha = runif(n_iter, 0.5, 5),                      # Stronger L1 regularization
  scale_pos_weight = runif(n_iter, 0.8, 1.2)          # Keep class balance flexible
)

best_rmse <- Inf
best_mae <- Inf
best_model <- NULL
best_predictions <- NULL

for (i in 1:n_iter) {
  params <- param_grid[i, ]
  model <- xgboost(
    data = train_matrix,
    label = train_labels,
    weight = train_weights,
    max_depth = params$max_depth,
    eta = params$eta,
    nrounds = params$nrounds,
    min_child_weight = params$min_child_weight,
    subsample = params$subsample,
    colsample_bytree = params$colsample_bytree,
    gamma = params$gamma,
    lambda = params$lambda,
    alpha = params$alpha,
    scale_pos_weight = params$scale_pos_weight,
    objective = "reg:squarederror",
    verbose = 0
  )
  
  pred <- predict(model, test_matrix)
  pred_exp <- exp(pred)  # Un-log the predictions
  
  mae <- mean(abs(pred_exp - test_data$next_batters_faced)) # MAE on original scale
  rmse <- sqrt(mean((pred_exp - test_data$next_batters_faced)^2))  # RMSE on original scale
  
  if (rmse < best_rmse) {  
    best_mae <- mae
    best_rmse <- rmse
    best_model <- model
    best_predictions <- data.frame(actual = test_data$next_batters_faced, predicted = pred_exp)
  }
  
}

# Print Best RMSE
print(paste("Best MAE:", best_mae))
print(paste("Best RMSE:", best_rmse))
# View Best Predictions (already un-logged)
view(best_predictions)

# Save predictions alongside the test data
results <- test_data %>%
  select(pitcher, year, stretch_id, next_batters_faced) %>%  # Keep relevant columns
  mutate(predicted_batters_faced = best_predictions$predicted)

# View final results
view(results)
avg_diff <- results %>%
  filter(predicted_batters_faced >= 500) %>%
  mutate(perc_diff = abs((next_batters_faced - predicted_batters_faced) / predicted_batters_faced) * 100) %>%
  summarise(avg_perc_diff = mean(perc_diff, na.rm = TRUE))
avg_diff
avg_diff <- as.numeric(avg_diff)
results$predicted_batters_faced <- (results$predicted_batters_faced * (1+(avg_diff/100)))
view(results)
# Plot Actual vs Predicted
ggplot(data = best_predictions, aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "solid", color = "red") + 
  ggtitle("Predicted vs Actual Batters Faced") +
  xlab("Actual Batters Faced") +
  ylab("Predicted Batters Faced") +
  theme_bw()

# save model
xgb.save(best_model, "batters_faced_xgb.model")

# Extract feature names from your training data
feature_names <- colnames(train_matrix)

# Extract feature importance with feature names
importance_matrix <- xgb.importance(feature_names = feature_names, model = best_model)

# Plot the feature importance
xgb.plot.importance(importance_matrix, 
                    measure = "Gain", 
                    main = "Feature Importances", 
                    rel_to_first = TRUE, 
                    xlab = "Relative Importance")
####################################################################################################
# Verify that 2023 data exists
df_season <- calculate_season_stats(df)
df_2023 <- df_season %>% filter(year == 2023)
df_2023 <- df_2023 %>%
  mutate(is_season_level = TRUE) 
df_2023 <- df_2023 %>%
  mutate(sample_weight = ifelse(is_season_level, 5, 1))
# Align columns for prediction
train_features <- colnames(train_matrix)
predict_features <- colnames(df_2023 %>% select(-c(pitcher, year, batters_faced)))

# Convert to matrix format for prediction
predict_matrix_2023 <- as.matrix(df_2023 %>% select(all_of(train_features)))

# Handle NA and infinite values in prediction data
predict_matrix_2023[is.na(predict_matrix_2023)] <- 0
predict_matrix_2023[is.infinite(predict_matrix_2023)] <- 0

# Predict using the trained model
predicted_bf_2024_log <- predict(best_model, predict_matrix_2023)

# Un-log the predictions
predicted_bf_2024 <- exp(predicted_bf_2024_log)

# Save and view results
results_2024 <- data.frame(
  pitcher = df_2023$pitcher,
  predicted_batters_faced_2024 = predicted_bf_2024
)
results_2024$predicted_batters_faced_2024 <- results_2024$predicted_batters_faced_2024 * avg_diff
view(results_2024)

ggplot(data = results_2024, aes(x = predicted_batters_faced_2024)) +
  geom_histogram() +
  ggtitle("Distribution of Predicted Batters Faced") +
  labs(x = "Predicted BF",
       y = "Count") +
  theme_bw()


bf2023 <- ggplot(data = df_2023, aes(x = batters_faced)) +
  geom_density(fill = "blue", alpha = 0.4) +
  ggtitle("Distribution of 2023 Batters Faced") +
  labs(x = "2023 Batters Faced",
       y = "Density") +
  theme_bw()


ggsave("2023_batters_faced.png", plot = bf2023, width = 8, height = 6, dpi = 300)


batter_stats <- df %>%
  group_by(batter,year) %>%
  summarise(
    games_played = n_distinct(game_pk),
    plate_appearances = sum(events %in% c(
      "strikeout", "field_out", "walk", "force_out", "sac_fly", "single",
      "hit_by_pitch", "double", "grounded_into_double_play", "home_run",
      "fielders_choice", "field_error", "triple", "strikeout_double_play",
      "fielders_choice_out", "double_play", "catcher_interf", "sac_bunt"
    ), na.rm = TRUE),
    at_bats = sum(events %in% c(
      "single", "double", "triple", "home_run", "strikeout", "field_out",
      "grounded_into_double_play", "fielders_choice", "field_error",
      "other_out", "strikeout_double_play", "fielders_choice_out",
      "double_play"
    ), na.rm = TRUE),
    singles = sum(events == "single", na.rm = TRUE),
    doubles = sum(events == "double", na.rm = TRUE),
    triples = sum(events == "triple", na.rm = TRUE),
    home_runs = sum(events == "home_run", na.rm = TRUE),
    hits = singles + doubles + triples + home_runs,
    walks = sum(events == "walk", na.rm = TRUE),
    strike_outs = sum(events == "strikeout", na.rm = TRUE) + sum(events == "strikeout_double_play", na.rm = TRUE), 
    hit_by_pitch = sum(events == "hit_by_pitch", na.rm = TRUE),
    sacrifice_hits = sum(events == "sac_bunt", na.rm = TRUE),
    sacrifice_flies = sum(events == "sac_fly", na.rm = TRUE),
    tb = sum(singles + (2 * doubles) + (3 * triples) + (4 * home_runs)),
    ba = hits / at_bats,
    slg = tb / at_bats,
    obp = (hits + walks + hit_by_pitch) /
      (plate_appearances - sacrifice_hits),
    ops = slg + obp,
    avg_ev = mean(launch_speed, na.rm = TRUE),
    avg_la = mean(launch_angle, na.rm = TRUE),
    bb_perc = walks/plate_appearances,
    k_perc = strike_outs/plate_appearances,
    xba = mean(estimated_ba_using_speedangle, na.rm = TRUE),
    woba = mean(woba_value, na.rm = TRUE),
    xwobacon = mean(estimated_woba_using_speedangle, na.rm = TRUE),
    hard_hit_balls = sum(launch_speed >= 95 & description != "foul", na.rm = TRUE),
    bbe = sum(events %in% c(
      "single", "double", "triple", "home_run", "field_out",
      "grounded_into_double_play", "fielders_choice", "field_error",
      "other_out", "strikeout_double_play", "fielders_choice_out",
      "double_play"
    ), na.rm = TRUE),
    hard_hit_rate = ifelse(bbe > 0, hard_hit_balls / bbe, 0),
    run_exp = mean(delta_run_exp, na.rm = TRUE),
  ) %>%
  ungroup()

bs_2023 <- batter_stats %>% filter(year == 2023)


pa2023 <- ggplot(data = bs_2023, aes(x = plate_appearances)) +
  geom_density(fill = "blue", alpha = 0.4) +
  ggtitle("Distribution of 2023 Plate Appearances") +
  labs(x = "2023 Plate Appearances",
       y = "Density") +
  theme_bw()

ggsave("2023_plate_appearances.png", plot = pa2023, width = 8, height = 6, dpi = 300)

predbf <- read_csv("predicted 2024 batters.csv")
predpa <- read_csv("PA_24_Predictions.csv")

summary_2023 <- summary(df_2023$batters_faced)
summary_2024 <- summary(predbf$predicted_batters_faced_2024)


summary_df <- data.frame(
  Statistic = c("Min", "1st Quartile", "Median", "Mean", "3rd Quartile", "Max"),
  `2023 Batters Faced` = as.numeric(summary_2023[c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")]),
  `2024 Predicted Batters Faced` = as.numeric(summary_2024[c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")])
)

library(gt)

bf_gt <- summary_df %>%
  gt() %>%
  tab_header(
    title = "Comparison of Batters Faced (2023 vs. 2024 Predictions)"
  ) %>%
  fmt_number(columns = 2:3, decimals = 2) %>%
  cols_label(
    Statistic = "Statistic",
    `X2023.Batters.Faced` = "2023",
    `X2024.Predicted.Batters.Faced` = "2024 (Predicted)"
  ) %>%
  tab_options(
    table.font.size = "medium",
    column_labels.font.weight = "bold"
  )

gtsave(bf_gt, "batters_faced_summary.png")

summary2023 <- summary(bs_2023$plate_appearances)
summary2024 <- summary(predpa$Plate_Appearances)

summarydf <- data.frame(
  Statistic = c("Min", "1st Quartile", "Median", "Mean", "3rd Quartile", "Max"),
  `2023 Plate Appearances` = as.numeric(summary2023[c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")]),
  `2024 Predicted Plate Appearances` = as.numeric(summary2024[c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")])
)

pa_gt <- summarydf %>%
  gt() %>%
  tab_header(
    title = "Comparison of Plate Appearances (2023 vs. 2024 Predictions)"
  ) %>%
  fmt_number(columns = 2:3, decimals = 2) %>%
  cols_label(
    Statistic = "Statistic",
    `X2023.Plate.Appearances` = "2023",
    `X2024.Predicted.Plate.Appearances` = "2024 (Predicted)"
  ) %>%
  tab_options(
    table.font.size = "medium",
    column_labels.font.weight = "bold"
  )

gtsave(pa_gt, "plate_appearances_summary.png")

view(summarydf)
