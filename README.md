# APRAU - Machine Learning Framework (ISEP 2025/2026)

This project was developed within the scope of the **Machine Learning (APRAU)** course for the Master’s in Data Engineering at **ISEP**. The core objective focused on the execution of a comprehensive and rigorous end-to-end Data Science pipeline. This methodological framework spanned from initial data ingestion and exhaustive exploratory analysis—to identify underlying patterns, correlations, and anomalies—to sophisticated feature engineering aimed at refining and optimizing model inputs. Furthermore, the study involved the benchmarking, and validation of diverse predictive models, specifically tailored to address both continuous regression challenges and complex multi-class classification problems within a high-dimensional environment.

## Project Overview and Objectives

The study focuses on a comprehensive dataset consisting of musical metadata and acoustic features (`group_15.csv`). The dataset comprises 3,000 observations and 49 independent variables, representing diverse characteristics such as rhythmic patterns, spectral energy, and popularity metrics. Working with this data presented several real-world challenges; notably, it required robust missing value imputation to preserve dataset integrity and advanced noise reduction to handle inherent signal-to-noise inconsistencies. Furthermore, addressing high levels of feature collinearity was critical, as overlapping information between acoustic indicators could otherwise lead to unstable coefficient estimates and reduced model interpretability.

The analytical goals are twofold:

1. **Regression Analysis**: Estimating the continuous variable `target_regression` utilizing both simple and multi-linear modeling techniques. This objective aims to quantify the predictive relationship between musical attributes and a target value, allowing for a precise evaluation of how specific features influence overall trends and identifying which acoustic signatures carry the most significant weight.
2. **Classification Analysis**: Categorizing instances into specific classes within the `target_class` variable. This involves exploring complex decision boundaries through various supervised learning algorithms, ranging from traditional linear discriminants to non-parametric models. The focus is on determining the most effective classifier for this multi-class problem while ensuring the model generalizes well to unseen data.

## Technical Implementation

### 1. Exploratory Data Analysis (EDA) & Pre-processing

Detailed in `Projeto_1.ipynb`, this phase ensured data integrity and established a statistical foundation for modeling:

- **Data Cleaning**: Systemic identification and handling of null values and duplicate records to ensure a clean training set.
- **Statistical Profiling**: Analysis of variable distributions using histograms and boxplots to detect outliers and skewness that could bias linear models.
- **Multicollinearity Assessment**: Implementation of correlation matrices to detect redundant features (e.g., the high variance inflation between `signal_strength` and `signal_power`), leading to a more streamlined and interpretable feature set.

### 2. Predictive Modeling

The modeling phase, documented in `Projeto_2.ipynb`, involved benchmarking several architectures to optimize predictive power and evaluate the trade-offs between model complexity and interpretability. By testing a wide spectrum of algorithms, the project aimed to identify the most robust solution for both the linear nature of regression and the potentially non-linear boundaries of the classification task.

### Regression Modeling

- **Ordinary Least Squares (OLS)**: Establishing baseline simple and multiple linear regression models. These baselines provided a crucial reference point for assessing the impact of individual features—such as `artists_avg_popularity`—on the target variable, helping to separate signal from noise early in the modeling process.
- **Model Optimization**: This involved an iterative refinement process through detailed residual analysis to ensure the validity of the OLS assumptions. Special attention was paid to checking for homoscedasticity (constant variance of errors) and normality of residuals. Furthermore, the systematic evaluation of p-values and F-statistics ensured that only statistically significant independent variables remained in the final model, thereby enhancing its predictive stability and reducing the risk of overfitting.

### Classification Frameworks

- **Discriminant Analysis**: A comparative study between Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA). This comparison allowed for an evaluation of whether the classes shared a common covariance matrix (LDA) or required unique, individual covariance structures (QDA), providing insights into the inherent geometric separation of the music categories.
- **Regularized Logistic Regression**: Implementation of L1 (Lasso), L2 (Ridge), and Elastic Net penalties. These regularization techniques were vital for handling the high-dimensional nature of the dataset; by shrinking the coefficients of less relevant features, the models performed automated feature selection, significantly mitigating the risk of overfitting in the presence of correlated predictors.
- **Support Vector Machines (SVM)**: Comprehensive testing across multiple Kernels, including Linear, Radial Basis Function (RBF), and Polynomial. This approach focused on finding the optimal separating hyperplane in a high-dimensional feature space, effectively transforming the input data to identify non-linear boundaries that distinguish different musical classes with high precision.
- **Generalized Additive Models (GAMs)**: Leveraging smoothing splines to capture complex, non-linear relationships. GAMs offered a sophisticated middle ground between the rigidity of linear models and the "black-box" nature of deep learning, allowing for flexible, non-parametric shapes for individual features while maintaining an additive, interpretable structure.

### 3. Dimensionality Reduction

**Principal Component Analysis (PCA)** was strategically integrated into the pipeline to address the "curse of dimensionality" inherent in a 49-variable dataset. By transforming the original feature space into a new set of orthogonal principal components, the project managed to compress the data while retaining over 90% of the total variance. This transformation not only improved computational efficiency and reduced training times across all models but also enhanced model generalization by eliminating noise and focusing on the directions of maximum variance within the acoustic metadata.

## Key Findings and Performance

- **Feature Influence and Predictive Drivers**: The rigorous statistical analysis identified `artists_avg_popularity` as the most significant predictor for the regression target. This finding suggests that social and metadata-driven factors, such as artist reputation and market reach, act as primary drivers in the dataset’s underlying patterns, often outweighing purely acoustic or technical features. The high statistical significance of this variable underscores the importance of contextual metadata in enhancing the predictive power of music-related models.
- **Classification Accuracy and Model Convergence**: The developed classification models achieved exceptional performance metrics across the board. F1-Macro and Accuracy scores consistently exceeded 90%, demonstrating high reliability in distinguishing between the various music categories. This level of accuracy remained stable across different architectures—from Regularized Logistic Regression to SVM—indicating that the dataset possesses well-defined class separations that were effectively captured by the chosen feature engineering approach.
- **Algorithmic Robustness and Non-Linear Adaptation**: Generalized Additive Models (GAMs) demonstrated superior flexibility and robustness, particularly when compared to standard parametric models. By utilizing smoothing splines, GAMs were able to outperform linear baselines in complex scenarios where the relationship between acoustic features (such as spectral energy) and the target class exhibited significant non-linear behavior. This adaptability allowed for a more nuanced fit, reducing bias without significantly increasing the risk of overfitting, thus providing a deeper understanding of the non-linear boundaries within the musical metadata.

## Repository Structure

- **`Projeto_1.ipynb` (Phase I: EDA and Data Preparation)**: This notebook serves as the project's analytical foundation, containing the full data cleaning pipeline. It documents the systematic handling of missing values through imputation, the removal of duplicate entries, and the mitigation of outliers. Additionally, it features an array of exploratory visualizations—including density plots and correlation heatmaps—that provide a granular understanding of the descriptive statistics and the intrinsic relationships within the feature space.
- **`Projeto_2.ipynb` (Phase II: Predictive Modeling and Validation)**: A detailed technical report focusing on the implementation and benchmarking of machine learning algorithms. This notebook covers the end-to-end training process, from hyperparameter tuning aimed at maximizing predictive accuracy to the use of rigorous cross-validation techniques. These validation methods ensure that performance metrics, such as RMSE and F1-scores, are reliable and that the models maintain high generalization capabilities across different data subsets.
- **`group_15.csv` (Primary Dataset)**: The core source file containing the 3,000 raw instances analyzed throughout the study. This CSV file includes the complete set of 49 variables, encompassing both the audio features and the labels required for the supervised learning tasks. It serves as the definitive ground truth for evaluating and benchmarking the effectiveness of every regression and classification strategy implemented in the project notebooks.

## Authors

- **David Tavares Mendonça**
- **Flávio Ferreira**
