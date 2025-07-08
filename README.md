# Product Categorisation using Machine Learning

## Overview

This project applies supervised machine learning algorithms to automate product category classification in an e-commerce dataset. The goal is to improve product organisation, optimise operations, and enhance user experience by automatically predicting categories (e.g., Furniture, Office Supplies, Technology) based on product and sales features.

## Dataset

The dataset (`Sales.csv`) was sourced from Data.world and contains over 9,000 rows and 20 columns, including:

* Order details (e.g., quantity, sales, profit)
* Customer and regional information
* Product details (category, subcategory)
* Dates and preparation time

The dataset was extensively cleaned, including outlier removal using Z-score, categorical encoding (One-Hot and Label Encoding), scaling, and feature engineering

## Machine Learning Algorithms

### 1️⃣ Support Vector Machine (SVM) with Polynomial Kernel

* **Description**: SVM is a supervised learning algorithm that finds the optimal hyperplane to separate classes. A polynomial kernel enables handling non-linear relationships by mapping features into a higher-dimensional space.
* **Implementation**: `sklearn.svm.SVC` with `kernel='poly'`, degree 3, `coef0=1`, and `C=5`, integrated into a pipeline with `StandardScaler`.
* **Performance**:

  * Training score: 0.716
  * Test score: 0.718
  * Accuracy: 0.718
* **Why chosen**: Effective at capturing complex, non-linear feature interactions while maintaining good generalisation.

### 2️⃣ Neural Network Classifier (MLPClassifier)

* **Description**: A Multi-Layer Perceptron (MLP) neural network mimics biological neurons to learn complex patterns through forward and backpropagation.
* **Implementation**: `sklearn.neural_network.MLPClassifier` with `max_iter=900`.
* **Performance**:

  * Training score: 0.749
  * Test score: 0.719
  * Accuracy: 0.719
* **Why chosen**: Excellent for capturing non-linear relationships and capable of modelling complex data distributions.

### 3️⃣ Decision Tree Classifier

* **Description**: A decision tree splits data into branches using feature-based decision rules, resulting in easy-to-interpret tree structures.
* **Implementation**: `sklearn.tree.DecisionTreeClassifier` with a maximum depth of 3 (initial model) and further hyperparameter tuning.
* **Performance**:

  * Training score: 0.673
  * Test score: 0.693
  * Accuracy: 0.693
* **Why chosen**: Provides a transparent model that is easy to interpret and visualise.

## Model Tuning

All models were optimised using hyperparameter tuning:

* **Grid Search**: Exhaustive search over specified parameter grids.
* **Randomized Search**: Sampling-based search for faster exploration of parameter space.

### Example Hyperparameters

* **SVM**: `C`, `gamma`, `kernel`
* **MLPClassifier**: `hidden_layer_sizes`, `activation`, `learning_rate`, `alpha`
* **Decision Tree**: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`

## Evaluation

Models were evaluated using:

* **Multilabel Confusion Matrices**: Provided class-level performance insights.
* **Metrics**: Accuracy, True Positive Rate (Recall), True Negative Rate, F1 Score, Precision.

The Neural Network achieved the highest accuracy after tuning (up to 0.745), followed closely by SVM and Decision Tree.

## Visualisations

* Correlation heatmaps
* Box plots (before and after outlier removal)
* Pie charts for categorical distributions
* Sales trends and category-wise sales charts

## Conclusion

The Neural Network classifier was the most effective model, closely followed by the SVM with a polynomial kernel. The Decision Tree classifier provided valuable interpretability and quick insights.
