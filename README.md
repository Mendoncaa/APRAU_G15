# APRAU - Machine Learning (ISEP 2025/2026)

This repository contains the projects developed for the **Machine Learning (APRAU)** course, part of the Master's in Data Engineering at **ISEP**.

The main objective is to apply and analyze various Machine Learning methods on a real dataset, covering everything from initial exploration to the implementation of complex regression and classification models.

## About the Project

The work is based on a dataset of audio features and music metadata (`group_15.csv`). The project is divided into two main areas:
1. **Regression**: Predicting the continuous variable `target_regression`.
2. **Classification**: Identifying the correct class in the `target_class` variable (Multiclass).

## Repository Structure

* `Projeto_1.ipynb`: Focused on **Exploratory Data Analysis (EDA)** and Data Cleaning. Includes handling of missing values, removal of duplicates, univariate/bivariate analysis, and multicollinearity study.
* `Projeto_2.ipynb`: Application of predictive methods:
    * **Regression**: Linear Regression (Simple and Multiple) with different feature selections.
    * **Classification**: Logistic Regression (with L1, L2, and Elastic Net penalties), LDA, QDA, SVM (Linear, RBF, Poly Kernels), and GAMs (Generalized Additive Models).
    * **Dimensionality Reduction**: Application of PCA (Principal Component Analysis).

## Key Findings

* **EDA**: The variable `artists_avg_popularity` was identified as having the highest individual correlation with the regression target.
* **Feature Engineering**: Redundant variables (e.g., `signal_strength` vs `signal_power`) were removed to reduce multicollinearity and improve model stability.
* **Performance**: Classification models achieved F1-Macro and Accuracy metrics above 90%, with GAMs standing out for their effectiveness in identifying specific classes.

