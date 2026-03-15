# Insurance Claim Approval Classification using Machine Learning

This project develops and evaluates machine learning models to predict **insurance claim approval outcomes** using healthcare, demographic, and insurance-related variables from the **Medical Expenditure Panel Survey (MEPS)** dataset.

The study compares multiple classification models and applies careful preprocessing, including **data leakage removal**, to ensure realistic and reliable model evaluation.

---

## Project Objective

The main goal of this project is to investigate whether machine learning models can accurately predict insurance claim approval decisions and identify which model performs best for this task.

This project focuses on:

- Building a binary classification pipeline for claim approval prediction
- Comparing the performance of multiple machine learning models
- Preventing data leakage for trustworthy evaluation
- Analysing feature importance and prediction behaviour
- Demonstrating how machine learning can support insurance decision systems

---

## Research Questions

1. Can machine learning models accurately predict insurance claim approval outcomes?
2. Which classification algorithm performs best among **Support Vector Machine (SVM)**, **Gradient Boosting**, and **Neural Network**?
3. Which features most strongly influence claim approval predictions?
4. How does strict preprocessing and leakage removal improve model reliability?

---

## Dataset

**Dataset Name:** Medical Expenditure Panel Survey (MEPS)  
**Source:** Agency for Healthcare Research and Quality (AHRQ), U.S. Department of Health and Human Services  
**Type:** Structured tabular dataset  
**Link:** https://meps.ahrq.gov/data_stats/download_data_files.jsp

### Dataset Characteristics

The dataset contains healthcare and insurance-related information, including:

- Demographic attributes
- Insurance coverage details
- Socio-economic indicators
- Healthcare utilisation variables
- Medical expenditure-related variables

In this project, selected variables were extracted from the MEPS file and renamed for easier interpretation.

---

## Selected Features

The notebook uses the following core variables:

- **AGE**
- **SEX**
- **RACE**
- **INCOME_PCT_FPL**
- **INSURANCE_TYPE**
- **TOTAL_EXPENDITURE**
- **EMPLOYED**
- **EDUCATION**
- **REGION**
- **NUM_VISITS**

---

## Target Variable Engineering

A binary target variable called **`CLAIM_APPROVED`** was created.

The target was defined using insurer-related payment/expenditure logic to simulate whether a claim was approved. This transforms the problem into a **binary classification task**:

- `1` = Claim approved
- `0` = Claim not approved

---

## Data Preprocessing

The notebook includes an end-to-end preprocessing pipeline designed to improve data quality and prevent unrealistic model performance.

### Steps performed

- Imported and loaded the raw MEPS `.dta` dataset
- Selected claim-relevant variables from the original file
- Renamed columns for readability
- Replaced MEPS negative codes (such as `-1`, `-7`, `-8`, `-9`) with missing values
- Removed duplicate rows
- Dropped rows with excessive missing values
- Imputed missing numerical values using the **median**
- Imputed missing categorical values using the **mode**
- Engineered the binary target variable **`CLAIM_APPROVED`**
- Performed **data leakage removal** by dropping variables directly revealing outcomes
- Applied encoding where needed
- Split the dataset into training and testing sets

### Leakage Control

A key strength of this project is the removal of leakage-related variables.  
Variables containing patterns such as:

- `EXP`
- `PAY`
- `CHG`
- `CHARGE`
- `PMT`
- `PRPAY`
- `MCDPAY`
- `PRVPAY`

were removed because they can directly reveal the target outcome and lead to artificially inflated performance.

---

## Exploratory Data Analysis (EDA)

The notebook includes exploratory analysis to understand the dataset before modelling.

### EDA covered

- Class distribution of claim approval
- Age distribution by approval class
- Relationship between insurance type and approval
- Approval rate by income category
- Distribution of expenditure using log transformation

These visualisations help identify patterns in healthcare utilisation, demographic variation, and approval outcomes.

---

## Models Implemented

The following machine learning models were implemented and evaluated:

### 1. Support Vector Machine (SVM)
Used as a strong baseline classifier for structured data classification.

### 2. Gradient Boosting Classifier
Used to capture non-linear relationships and improve predictive performance through boosting.

### 3. Deep Neural Network
A feedforward neural network built using **TensorFlow/Keras** with:

- Input layer
- Dense hidden layers
- Dropout regularisation
- Sigmoid output layer for binary classification

---

## Model Evaluation Metrics

The models were evaluated using the following classification metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**

These metrics provide a more complete evaluation than accuracy alone, especially for classification tasks with class imbalance.

---

## Model Performance

Based on the notebook output, the model comparison results are:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|---------:|----------:|-------:|---------:|--------:|
| Gradient Boosting | 0.8868 | 0.8907 | 0.8868 | 0.8886 | 0.9439 |
| Neural Network | 0.8835 | 0.8833 | 0.8835 | 0.8834 | 0.9432 |
| SVM | 0.8470 | 0.8215 | 0.8470 | 0.8206 | 0.9184 |

### Best Model
The **Gradient Boosting model** achieved the best overall performance, with the highest:

- Accuracy
- Precision
- F1 Score
- ROC-AUC

This suggests that ensemble methods are highly effective for insurance claim approval classification on structured healthcare data.

---

## Feature Importance

Feature importance analysis was performed using the **Gradient Boosting model**.

The results indicate that **healthcare visits** were the most influential predictor, followed by other variables such as:

- Age
- Insurance type
- Race
- Education
- Region

This shows that both **healthcare utilisation** and **demographic factors** play an important role in predicting claim approval outcomes.

---

## Actual vs Predicted Analysis

The notebook also includes a comparison of **actual vs predicted values** for the Neural Network model.

This helps to:

- Inspect prediction correctness
- Review approval probabilities
- Understand where the model makes correct and incorrect classifications

A sample output table is included in the notebook showing:

- Actual class
- Predicted class
- Probability of approval
- Whether the prediction is correct

---

## Visualisations Included

The notebook contains the following visual outputs:

- EDA plots
- Confusion matrices
- ROC curves
- Model comparison charts
- Feature importance plot
- Actual vs predicted sample table

These visualisations make the results easier to interpret and communicate.

---

## Project Workflow

The end-to-end workflow of the project is:

1. Load raw MEPS dataset  
2. Select relevant variables  
3. Clean and preprocess the data  
4. Engineer target variable  
5. Remove leakage-related features  
6. Perform exploratory data analysis  
7. Split data into train and test sets  
8. Train machine learning models  
9. Evaluate model performance  
10. Compare models  
11. Analyse feature importance  
12. Review prediction behaviour  

---

## Key Findings

- Machine learning models can effectively predict insurance claim approval outcomes
- Ensemble methods outperform the baseline SVM model
- Gradient Boosting achieved the strongest overall results
- Removing leakage-related variables is critical for realistic evaluation
- Healthcare utilisation variables appear to be highly predictive
- The project demonstrates the practical value of ML in insurance decision support

---

## Limitations

- The target variable is engineered rather than directly observed as a real insurer approval label
- The project uses selected variables rather than the entire MEPS feature space
- Additional tuning and validation could further improve model robustness

---

## Future Work

Possible future improvements include:

- Hyperparameter tuning using Grid Search or Random Search
- k-fold cross-validation
- Testing advanced ensemble models such as XGBoost or LightGBM
- More detailed feature engineering
- Fairness analysis across demographic groups
- Deployment as a real-time decision-support system using Azure Machine Learning or FastAPI

---

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
