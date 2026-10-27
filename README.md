# Diabetes_BRFSS_ML

R script for diabetes prediction using the BRFSS 2015 balanced dataset.

Compares **Random Forest**, **XGBoost**, and **LightGBM**.

## What it does
- Table 1 (descriptive statistics)
- 70/30 train-test split
- Model comparison (Accuracy, Precision, Recall, F1, AUC)
- ROC curves
- Feature importance and SHAP plot for the best model

## How to run

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

2. Create a folder called `data` and put the CSV inside it.  
   Name the file **`BRFSS.csv`**

3. Install packages in R:

```r
install.packages(c(
  "readr", "dplyr", "tidyr", "ggplot2", "caret",
  "ranger", "xgboost", "lightgbm", "pROC",
  "tableone", "openxlsx"
))

## How to cite

If you use this script, please cite it as:

Ashraf A. Noah (2026). Diabetes_BRFSS_ML: R script for diabetes prediction using BRFSS 2015. 
GitHub. https://github.com/Ashraf-AlamEldin/Diabetes_BRFSS_ML
