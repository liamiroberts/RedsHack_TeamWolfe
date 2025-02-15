library(tidyverse)
library(dplyr)
library(readxl)
library(ggplot2)
library(xgboost)
library(caret)
library(Matrix)
library(xgboost)
setwd('/Users/jamesonbodenburg/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/School/SAL 358/reds-hackathon-2025')

savant_data <- read_csv("savant_data_2021_2023.csv")
players <- read_csv('lahman_people.csv')

names(savant_data)

savant_data$year <- format(savant_data$game_date, "%Y")

batter_stats <- savant_data %>%
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
    avg_order_spot = mean(ifelse(at_bat_number%%9>0, at_bat_number%%9, 9))
  ) %>%
  ungroup()

games_missed <- savant_data %>% 
  group_by(game_pk, batter, year) %>% 
  summarize(batter=unique(batter),
            game_date = first(game_date),
            home_team = first(home_team),
            away_team = first(away_team),
            batter_team = unique(ifelse(inning_topbot=="Bot", home_team, away_team)),
            .groups="drop")

games_missed <- games_missed %>%
  left_join(players, by = c("batter" = "player_mlb_id"))

games_missed$age <- as.numeric(difftime(games_missed$game_date, games_missed$birthDate, units = "days")) / 365.25  

batter_teams <- games_missed %>% 
  group_by(batter, year, batter_team) %>% 
  summarize(batter = first(batter),
            batter_team = first(batter_team))

team_games <- games_missed %>% 
  group_by(game_pk) %>% 
  reframe(home_team = first(home_team),
            away_team = first(away_team),
          year = first(year))

all_games <- team_games %>%
  pivot_longer(cols = c(home_team, away_team), names_to = "home_away", values_to = "batter_team")

batter_teams <- games_missed %>%
  distinct(batter, batter_team, year)

all_games <- all_games %>%
  left_join(batter_teams, by = c("batter_team", "year")) %>%  # Keep all team games
  filter(!is.na(batter)) 

all_games <- all_games %>%
  left_join(games_missed %>% select(batter, game_pk, game_date) %>% mutate(played = 1),
            by = c("batter", "game_pk")) %>%
  mutate(played = ifelse(is.na(played), 0, 1))


all_games <- all_games %>%
  arrange(batter, game_pk)



#Just used this for a histogram to compare actual games played with my results
real_data <- read_xlsx("example_data.xlsx")
hist(real_data$G)








####XGboost for games missed with each row being a batter####

# Step 1: Count total games missed per player per year
games_missed_by_year <- all_games %>%
  distinct(batter, year, game_pk, .keep_all = TRUE) %>%  # Keep only unique player-season-game pairs
  group_by(batter, year) %>%
  summarize(games_missed = min(sum(played == 0), 162),  # Cap missed games to 162 per season
            .groups = "drop") %>%
  pivot_wider(names_from = year, values_from = games_missed, names_prefix = "games_missed_") %>%
  replace_na(list(games_missed_2021 = 0, games_missed_2022 = 0, games_missed_2023 = 0))



total_missed_info <- all_games %>%
  arrange(batter, year, game_pk) %>%
  group_by(batter, year) %>%
  mutate(
    streak_id = cumsum(played == 1),  # Unique streak ID per player
    missed_streak = ifelse(played == 0, 1, 0)  # Identify missed games
  ) %>%
  group_by(batter, year, streak_id) %>%
  summarize(
    games_missed_in_streak = sum(missed_streak),  # Sum of missed games in the streak
    .groups = "drop"
  ) %>%
  filter(games_missed_in_streak >= 5) %>%  # Keep only streaks of 5+ missed games
  group_by(batter, year) %>%
  summarize(
    total_missed_streaks = n(),  # Count the number of streaks
    total_missed_days_in_streaks = sum(games_missed_in_streak),  # Sum missed days across streaks
    .groups = "drop"
  )


player_ages <- games_missed %>%
  select(batter, birthDate) %>%
  mutate(
    birthDate = as.Date(birthDate),  # Ensure it's in Date format
    age = floor(as.numeric(difftime(as.Date("2024-03-01"), birthDate, units = "days")) / 365.25)  # Calculate age
  ) %>%
  select(batter, age)


# Step 4: Combine all summaries into a single dataframe
player_summary <- games_missed_by_year %>%
  left_join(total_missed_info, by = "batter") %>%
  left_join(player_ages, by = "batter") %>%
  replace_na(list(total_missed_streaks = 0, total_missed_days_in_streaks = 0))


# Ensure each batter has only one row
player_summary <- player_summary %>% 
  distinct(batter, .keep_all = TRUE) %>% 
  select(-year)






###XGBoost

## This was the initial xgboost model but isn't the one I used (just keeping for reference). I used the final_xgb_model, which was tuned for parameters.



train_data <- player_summary %>%
  select(batter, age, total_missed_streaks, total_missed_days_in_streaks, 
         games_missed_2021, games_missed_2022, games_missed_2023)
# Ensure no missing values
train_data[is.na(train_data)] <- 0
# Define training matrix (features)
train_matrix <- train_data %>%
  select(-batter, -games_missed_2023) %>%  # Exclude target variable
  as.matrix()
# Define target variable (games missed in 2023 as a proxy for 2024)
train_target <- train_data$games_missed_2023
dtrain <- xgb.DMatrix(data = train_matrix, label = train_target)
# Train the model
xgb_model <- xgboost(
  data = dtrain,
  objective = "reg:squarederror",  # Regression model
  nrounds = 100,  # Number of boosting rounds
  max_depth = 6,   # Depth of trees
  eta = 0.1,       # Learning rate
  verbose = 1
)
predict_2024 <- player_summary %>%
  select(batter, age, total_missed_streaks, total_missed_days_in_streaks, 
         games_missed_2021, games_missed_2022, games_missed_2023)
# Ensure no missing values
predict_2024[is.na(predict_2024)] <- 0
train_features <- colnames(train_matrix)
# Ensure predict_2024 has the same column names
predict_matrix <- predict_2024 %>%
  select(all_of(train_features)) %>%  # Select only matching columns
  as.matrix()
# Convert to XGBoost DMatrix
dpredict <- xgb.DMatrix(data = predict_matrix)
# Predict 2024 games missed
predict_2024$predicted_games_missed_2024 <- predict(tuned_model$bestTune, dpredict)
# Limit the predictions to be between 0 and 162 (inclusive)
predict_2024$predicted_games_missed_2024 <- pmin(pmax(predict_2024$predicted_games_missed_2024, 0), 162)








####XGBoost validation####
# Calculate feature importance
importance_matrix <- xgb.importance(model = xgb_model)

# Plot variable importance
xgb.plot.importance(importance_matrix)





####Tuning####
tune_grid <- expand.grid(
  nrounds = c(100, 200),                 # number of boosting rounds (trees)
  eta = c(0.01, 0.1, 0.3),              # learning rate
  max_depth = c(3, 5, 7),                # max depth of trees
  min_child_weight = c(1, 5, 10),        # min sum of instance weight
  subsample = c(0.5, 0.7, 1),            # subsample ratio
  colsample_bytree = c(0.5, 0.7, 1),     # features to sample per tree
  gamma = c(0, 1)                        # regularization parameter
)

# Train the model with cross-validation
train_control <- trainControl(
  method = "cv",                         # Cross-validation
  number = 5,                             # 5-fold cross-validation
  verboseIter = TRUE,                     # Show progress
  allowParallel = TRUE                    # Allow parallel processing
)

# Train model using cross-validation
tuned_model <- train(
  games_missed_2023 ~ .,                       # Formula: predicting games missed
  data = train_data,                      # Your training data
  method = "xgbTree",                     # Using XGBoost
  trControl = train_control,              # Cross-validation settings
  tuneGrid = tune_grid                    # Hyperparameter grid
)

best_tuned_params <- tuned_model$bestTune

train_matrix <- train_data %>% 
  select(-batter) %>%   # Remove categorical column
  as.matrix()



# Train the XGBoost model using the best parameters
final_xgb_model <- xgboost(
  data = train_matrix,  
  label = train_data$games_missed_2023,  # Target variable
  nrounds = best_tuned_params$nrounds,
  max_depth = best_tuned_params$max_depth,
  eta = best_tuned_params$eta,
  gamma = best_tuned_params$gamma,
  colsample_bytree = best_tuned_params$colsample_bytree,
  min_child_weight = best_tuned_params$min_child_weight,
  subsample = best_tuned_params$subsample,
  objective = "reg:squarederror",  # Regression task
  verbose = 1
)

feature_cols <- c("age", "total_missed_streaks", "total_missed_days_in_streaks", 
                  "games_missed_2021", "games_missed_2022")

train_matrix <- as.matrix(train_data[, feature_cols])

predict_2024_matrix <- as.matrix(predict_2024[, feature_cols])

dpredict <- xgb.DMatrix(data = predict_2024_matrix)

predict_2024$predicted_games_missed_2024 <- predict(final_xgb_model, dpredict)
predict_2024$predicted_games_missed_2024 <- pmin(pmax(predict_2024$predicted_games_missed_2024, 0), 162)



library(writexl)
write_xlsx(predict_2024, "2024_predicted_games_missed.xlsx")


