# ============================================================
# Diabetes Prediction — BRFSS Diabetes Health Indicators
# Random Forest vs XGBoost vs LightGBM
# ============================================================

# ---- Load Libraries ----
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(ranger)
library(xgboost)
library(lightgbm)
library(pROC)
library(tableone)
library(openxlsx)

# ---- Paths (edit if needed) ----
# Put BRFSS.csv inside a folder called "data"
data_path <- "data/BRFSS.csv"
out_dir   <- "output"

# Create output folder if it doesn't exist
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Check if data file exists
if (!file.exists(data_path)) {
  stop("Data file not found!\n",
       "Please download the dataset and save it as: data/BRFSS.csv\n",
       "Link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")
}

# ---- Load & Prepare Data ----
set.seed(42)
df <- read_csv(data_path, show_col_types = FALSE)

stopifnot("Diabetes_binary" %in% names(df))

df <- df %>%
  mutate(
    Diabetes_binary = as.numeric(Diabetes_binary),
    y_factor = factor(Diabetes_binary, levels = c(0, 1), labels = c("No", "Yes"))
  )

cat("Rows:", nrow(df), "  Columns:", ncol(df), "\n")
print(table(df$y_factor))
cat("Missing values:", sum(is.na(df)), "\n\n")

# ---- Table 1: Descriptives ----
tab1_df <- df %>%
  mutate(
    HighBP = factor(HighBP, 0:1, c("No", "Yes")),
    HighChol = factor(HighChol, 0:1, c("No", "Yes")),
    CholCheck = factor(CholCheck, 0:1, c("No", "Yes")),
    Smoker = factor(Smoker, 0:1, c("No", "Yes")),
    Stroke = factor(Stroke, 0:1, c("No", "Yes")),
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack, 0:1, c("No", "Yes")),
    PhysActivity = factor(PhysActivity, 0:1, c("No", "Yes")),
    Fruits = factor(Fruits, 0:1, c("No", "Yes")),
    Veggies = factor(Veggies, 0:1, c("No", "Yes")),
    HvyAlcoholConsump = factor(HvyAlcoholConsump, 0:1, c("No", "Yes")),
    AnyHealthcare = factor(AnyHealthcare, 0:1, c("No", "Yes")),
    NoDocbcCost = factor(NoDocbcCost, 0:1, c("No", "Yes")),
    DiffWalk = factor(DiffWalk, 0:1, c("No", "Yes")),
    Sex = factor(Sex, 0:1, c("Female", "Male")),
    GenHlth = factor(GenHlth, 1:5, c("Excellent", "Very good", "Good", "Fair", "Poor")),
    Diabetes = y_factor
  )

vars <- c(
  "Age", "Sex", "BMI", "GenHlth", "Education", "Income",
  "HighBP", "HighChol", "CholCheck",
  "Smoker", "HvyAlcoholConsump", "PhysActivity", "Fruits", "Veggies",
  "Stroke", "HeartDiseaseorAttack", "DiffWalk",
  "MentHlth", "PhysHlth",
  "AnyHealthcare", "NoDocbcCost"
)

cat_vars <- c(
  "Sex", "GenHlth",
  "HighBP", "HighChol", "CholCheck",
  "Smoker", "HvyAlcoholConsump", "PhysActivity", "Fruits", "Veggies",
  "Stroke", "HeartDiseaseorAttack", "DiffWalk",
  "AnyHealthcare", "NoDocbcCost"
)

table1 <- CreateTableOne(
  vars = vars,
  strata = "Diabetes",
  data = tab1_df,
  factorVars = cat_vars,
  addOverall = TRUE
)

tab1_print <- print(table1, showAllLevels = TRUE, quote = FALSE, 
                    noSpaces = TRUE, printToggle = FALSE)

tab1_excel <- as.data.frame(tab1_print)
tab1_excel <- cbind(Variable = rownames(tab1_excel), tab1_excel)
rownames(tab1_excel) <- NULL

write.xlsx(tab1_excel, file.path(out_dir, "table1_descriptives.xlsx"))
cat("Table 1 saved.\n")

# ---- Train / Test Split (70/30) ----
idx <- createDataPartition(df$y_factor, p = 0.70, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

cat("\nTrain size:", nrow(train), " | Test size:", nrow(test), "\n")

feature_names <- setdiff(names(df), c("Diabetes_binary", "y_factor"))
x_train <- as.matrix(train[, feature_names])
x_test  <- as.matrix(test[, feature_names])
y_train_num <- train$Diabetes_binary
y_test_num  <- test$Diabetes_binary
y_train_fac <- train$y_factor
y_test_fac  <- test$y_factor

# ---- Metrics Helper ----
eval_binary <- function(y_true_fac, y_prob, model_name) {
  y_pred_fac <- factor(ifelse(y_prob >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(y_pred_fac, y_true_fac, positive = "Yes")
  auc_val <- as.numeric(auc(roc(as.numeric(y_true_fac == "Yes"), y_prob, quiet = TRUE)))
  
  tibble(
    model     = model_name,
    accuracy  = unname(cm$overall["Accuracy"]),
    precision = unname(cm$byClass["Precision"]),
    recall    = unname(cm$byClass["Recall"]),
    f1        = unname(cm$byClass["F1"]),
    auc       = auc_val
  )
}

# ---- SHAP Beeswarm Plot Function ----
plot_shap_beeswarm <- function(shap_mat, X, model_name, top_n = 12) {
  shap_df <- as.data.frame(shap_mat)
  feat_df <- as.data.frame(X)
  names(shap_df) <- colnames(X)
  names(feat_df) <- colnames(X)
  
  shap_long <- shap_df %>%
    mutate(id = row_number()) %>%
    pivot_longer(-id, names_to = "feature", values_to = "shap")
  
  feat_long <- feat_df %>%
    mutate(id = row_number()) %>%
    pivot_longer(-id, names_to = "feature", values_to = "value")
  
  plot_df <- shap_long %>%
    left_join(feat_long, by = c("id", "feature")) %>%
    group_by(feature) %>%
    mutate(
      mean_abs = mean(abs(shap)),
      value_scaled = as.numeric(scale(value))
    ) %>%
    ungroup()
  
  top_feats <- plot_df %>%
    distinct(feature, mean_abs) %>%
    arrange(desc(mean_abs)) %>%
    slice_head(n = top_n) %>%
    pull(feature)
  
  plot_df %>%
    filter(feature %in% top_feats) %>%
    mutate(feature = factor(feature, levels = rev(top_feats))) %>%
    ggplot(aes(shap, feature, color = value_scaled)) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey55") +
    geom_jitter(height = 0.22, width = 0, size = 0.35, alpha = 0.35) +
    scale_color_gradient2(
      low = "#3B4CC0", mid = "#F7F7F7", high = "#B40426",
      midpoint = 0, na.value = "grey70",
      name = "Feature value\n(low → high)"
    ) +
    labs(
      title = paste("SHAP Beeswarm —", model_name),
      x = "SHAP value (impact on log-odds of diabetes)",
      y = NULL
    ) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "right")
}

# ============================================================
# Models
# ============================================================

# ---- Random Forest ----
cat("\nTraining Random Forest...\n")
rf_fit <- ranger(
  dependent.variable.name = "y_factor",
  data            = train[, c(feature_names, "y_factor")],
  num.trees       = 500,
  mtry            = floor(sqrt(length(feature_names))),
  min.node.size   = 20,
  probability     = TRUE,
  importance      = "impurity",
  seed            = 42,
  num.threads     = max(1, parallel::detectCores() - 1)
)
rf_prob <- predict(rf_fit, data = test)$predictions[, "Yes"]
rf_metrics <- eval_binary(y_test_fac, rf_prob, "Random Forest")

# ---- XGBoost ----
cat("Training XGBoost...\n")
dtrain <- xgb.DMatrix(x_train, label = y_train_num)
dtest  <- xgb.DMatrix(x_test,  label = y_test_num)

xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = 4,
  eta              = 0.05,
  subsample        = 0.80,
  colsample_bytree = 0.80,
  min_child_weight = 5,
  nthread          = max(1, parallel::detectCores() - 1)
)

xgb_fit <- xgb.train(
  params  = xgb_params,
  data    = dtrain,
  nrounds = 300,
  evals   = list(train = dtrain, test = dtest),
  early_stopping_rounds = 30,
  verbose = 0
)
xgb_prob <- predict(xgb_fit, dtest)
xgb_metrics <- eval_binary(y_test_fac, xgb_prob, "XGBoost")

# ---- LightGBM ----
cat("Training LightGBM...\n")
lgb_train <- lgb.Dataset(data = x_train, label = y_train_num)
lgb_valid <- lgb.Dataset.create.valid(lgb_train, data = x_test, label = y_test_num)

lgb_params <- list(
  objective        = "binary",
  metric           = "auc",
  learning_rate    = 0.05,
  num_leaves       = 15,
  max_depth        = 4,
  feature_fraction = 0.80,
  bagging_fraction = 0.80,
  bagging_freq     = 1,
  min_data_in_leaf = 50,
  verbosity        = -1
)

lgb_fit <- lgb.train(
  params                = lgb_params,
  data                  = lgb_train,
  nrounds               = 300,
  valids                = list(test = lgb_valid),
  early_stopping_rounds = 30,
  verbose               = -1
)
lgb_prob <- predict(lgb_fit, x_test)
lgb_metrics <- eval_binary(y_test_fac, lgb_prob, "LightGBM")

# ---- Compare Models ----
results <- bind_rows(rf_metrics, xgb_metrics, lgb_metrics) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

cat("\n==================== TEST SET RESULTS ====================\n")
print(results)

# Confusion matrices
rf_pred  <- factor(ifelse(rf_prob  >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
xgb_pred <- factor(ifelse(xgb_prob >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
lgb_pred <- factor(ifelse(lgb_prob >= 0.5, "Yes", "No"), levels = c("No", "Yes"))

cat("\n--- Random Forest ---\n")
print(confusionMatrix(rf_pred, y_test_fac, positive = "Yes"))
cat("\n--- XGBoost ---\n")
print(confusionMatrix(xgb_pred, y_test_fac, positive = "Yes"))
cat("\n--- LightGBM ---\n")
print(confusionMatrix(lgb_pred, y_test_fac, positive = "Yes"))

# ---- ROC Plot ----
roc_rf  <- roc(y_test_num, rf_prob,  quiet = TRUE)
roc_xgb <- roc(y_test_num, xgb_prob, quiet = TRUE)
roc_lgb <- roc(y_test_num, lgb_prob, quiet = TRUE)

roc_df <- bind_rows(
  tibble(fpr = 1 - roc_rf$specificities,  tpr = roc_rf$sensitivities,  model = "Random Forest"),
  tibble(fpr = 1 - roc_xgb$specificities, tpr = roc_xgb$sensitivities, model = "XGBoost"),
  tibble(fpr = 1 - roc_lgb$specificities, tpr = roc_lgb$sensitivities, model = "LightGBM")
)

p_roc <- ggplot(roc_df, aes(fpr, tpr, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = 2, color = "grey50") +
  labs(title = "ROC Curves (Test Set)", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 13) +
  coord_equal()

ggsave(file.path(out_dir, "roc_comparison.png"), p_roc, width = 7, height = 6, dpi = 150)

# ---- Choose Winner ----
best_auc <- max(results$auc)
top_models <- results %>% filter(auc >= best_auc - 1e-12)
winner <- if (nrow(top_models) == 1) top_models$model[1] else top_models$model[which.max(top_models$f1)]
cat("\nBest model:", winner, "\n")

# ---- Feature Importance + SHAP for Winner ----
if (winner == "Random Forest") {
  imp <- tibble(
    feature    = names(rf_fit$variable.importance),
    importance = as.numeric(rf_fit$variable.importance)
  ) %>% arrange(desc(importance))
  
  p_imp <- imp %>%
    slice_head(n = 15) %>%
    mutate(feature = reorder(feature, importance)) %>%
    ggplot(aes(importance, feature)) +
    geom_col(fill = "#2C7FB8") +
    labs(title = "Random Forest — Impurity Importance", x = "Importance") +
    theme_minimal(base_size = 13)
  
} else if (winner == "XGBoost") {
  imp_mat <- xgb.importance(feature_names = feature_names, model = xgb_fit)
  imp <- as_tibble(imp_mat) %>%
    transmute(feature = Feature, importance = Gain) %>%
    arrange(desc(importance))
  
  p_imp <- imp %>%
    slice_head(n = 15) %>%
    mutate(feature = reorder(feature, importance)) %>%
    ggplot(aes(importance, feature)) +
    geom_col(fill = "#2C7FB8") +
    labs(title = "XGBoost — Gain Importance", x = "Gain") +
    theme_minimal(base_size = 13)
  
  # SHAP
  shap_n <- min(3000, nrow(x_test))
  shap_rows <- sample.int(nrow(x_test), shap_n)
  shap_x <- x_test[shap_rows, , drop = FALSE]
  shap_contrib <- predict(xgb_fit, xgb.DMatrix(shap_x), predcontrib = TRUE)
  shap_mat <- shap_contrib[, -ncol(shap_contrib), drop = FALSE]
  colnames(shap_mat) <- feature_names
  
  p_shap <- plot_shap_beeswarm(shap_mat, shap_x, "XGBoost")
  ggsave(file.path(out_dir, "shap_beeswarm.png"), p_shap, width = 8, height = 6.5, dpi = 150)
  
} else {  # LightGBM
  imp_mat <- lgb.importance(lgb_fit, percentage = TRUE)
  imp <- as_tibble(imp_mat) %>%
    transmute(feature = Feature, importance = Gain) %>%
    arrange(desc(importance))
  
  p_imp <- imp %>%
    slice_head(n = 15) %>%
    mutate(feature = reorder(feature, importance)) %>%
    ggplot(aes(importance, feature)) +
    geom_col(fill = "#2C7FB8") +
    labs(title = "LightGBM — Gain Importance", x = "Gain") +
    theme_minimal(base_size = 13)
  
  # SHAP
  shap_n <- min(3000, nrow(x_test))
  shap_rows <- sample.int(nrow(x_test), shap_n)
  shap_x <- x_test[shap_rows, , drop = FALSE]
  shap_contrib <- predict(lgb_fit, shap_x, type = "contrib")
  shap_mat <- shap_contrib[, -ncol(shap_contrib), drop = FALSE]
  colnames(shap_mat) <- feature_names
  
  p_shap <- plot_shap_beeswarm(shap_mat, shap_x, "LightGBM")
  ggsave(file.path(out_dir, "shap_beeswarm.png"), p_shap, width = 8, height = 6.5, dpi = 150)
}

ggsave(file.path(out_dir, "winner_importance.png"), p_imp, width = 7, height = 6, dpi = 150)

# ---- Save Results ----
write.xlsx(results, file.path(out_dir, "model_comparison_metrics.xlsx"))
write.xlsx(as.data.frame(imp), file.path(out_dir, "winner_feature_importance.xlsx"))

cat("\nAll files saved in the 'output' folder.\n")
cat("Done!\n")