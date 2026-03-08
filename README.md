# Insurance Claim Approval Classification using Machine Learning

## Project Overview

This project develops a machine learning system to predict whether an insurance claim will be **approved or rejected** based on customer, policy, and health-related attributes.

Insurance companies process thousands of claims every day, and manual decision-making can be time-consuming and inconsistent. By applying machine learning techniques, this project aims to assist insurance providers in automating claim approval predictions and identifying the most influential factors that affect approval outcomes.

The study compares several classification algorithms and evaluates their performance using multiple evaluation metrics.

---

# Project Objectives

The key objectives of this project are:

* To build machine learning models capable of predicting insurance claim approval outcomes.
* To compare the predictive performance of multiple classification algorithms.
* To identify the most influential features affecting insurance claim approval decisions.
* To evaluate models using standard classification metrics such as Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

---

# Dataset Description

The dataset contains information related to insurance policyholders, healthcare utilisation, and expenditure attributes.

Typical variables include:

* Age
* Gender
* Income level
* Health status
* Insurance coverage type
* Healthcare utilisation indicators
* Medical expenditure variables
* Demographic information

The target variable created for this project is:

**CLAIM_APPROVED**

Where:

* **1 = Claim Approved**
* **0 = Claim Rejected**

---

# Machine Learning Workflow

The project follows a structured machine learning pipeline:

### 1 Data Pre-processing

* Removal of leakage-related variables
* Handling missing values using median imputation
* Selection of numeric features
* Train–test split

### 2 Feature Engineering

* Identification of relevant attributes
* Removal of identifier variables
* Prevention of target leakage

### 3 Model Development

Four machine learning classification algorithms were implemented:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Neural Network (MLPClassifier)

### 4 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* ROC Curve Analysis

---

# Model Performance Comparison

| Model                | Accuracy   | AUC-ROC    | F1 Score   | Precision  | Recall     |
| -------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Random Forest        | 0.8828     | **0.9405** | 0.8867     | 0.8927     | 0.8828     |
| Gradient Boosting    | **0.8848** | 0.9394     | **0.8882** | **0.8933** | **0.8848** |
| Neural Network (MLP) | 0.8802     | 0.9378     | 0.8834     | 0.8880     | 0.8802     |
| Logistic Regression  | 0.8741     | 0.9364     | 0.8703     | 0.8679     | 0.8741     |

---

# Best Performing Model

The **Random Forest classifier** achieved the highest **ROC-AUC score (0.9405)**, indicating the strongest ability to distinguish between approved and rejected insurance claims.

Although Gradient Boosting slightly outperformed other models in Accuracy and F1 Score, Random Forest was selected as the best overall model due to its superior discriminative performance and robustness.

---

# Key Visualisations

The project includes several analytical visualisations such as:

* Model comparison bar charts
* ROC curve analysis
* Confusion matrix visualisation
* Feature importance ranking

These visualisations help interpret model behaviour and highlight important predictors influencing claim approval.

---

# Feature Importance Analysis

Feature importance analysis was performed using the Random Forest model to identify the variables that most strongly influence claim approval decisions.

This analysis helps insurance companies understand which attributes contribute most to claim approval outcomes and supports more transparent decision-making.

---

# Technologies and Tools Used

The following tools and technologies were used in this project:

Programming Language:

* Python

Data Processing:

* Pandas
* NumPy

Machine Learning:

* Scikit-Learn

Data Visualisation:

* Matplotlib
* Seaborn

Development Environment:

* Jupyter Notebook

Version Control:

* GitHub

