# Disease Prediction from Medical Data

## Overview

This project develops a machine learning system for estimating the likelihood of heart disease from structured medical data.

The project uses the **UCI Heart Disease — Cleveland dataset** and implements a complete machine learning pipeline:

```text
Medical Data
     ↓
Data Loading
     ↓
Data Validation
     ↓
Preprocessing
     ↓
Train/Test Split
     ↓
Machine Learning Models
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Disease Prediction
     ↓
Explainability
```

The main objective is to demonstrate how machine learning can be applied to structured medical data while keeping the workflow reproducible, modular, and understandable.

> **Important:** This project is intended for educational and research purposes. It is not a medical diagnostic system and must not be used to make real clinical decisions.

---

# Project Objectives

The project has several objectives:

* Load and validate structured medical data.
* Explore the characteristics of the dataset.
* Handle missing values using a preprocessing pipeline.
* Split the data into training and testing sets.
* Train multiple machine learning classification models.
* Compare the models using several evaluation metrics.
* Select the best-performing model.
* Save trained models for later use.
* Make predictions on new patient data.
* Provide interpretable explanations for model predictions.
* Organize the project using a reproducible Python structure.

---

# Dataset

## UCI Heart Disease Dataset

The project uses the **Cleveland subset of the UCI Heart Disease dataset**.

Official source:

https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset contains:

* **303 patient records**
* **13 input features**
* **1 target variable**

The original UCI target variable contains values from `0` to `4`.

For this project, the target is transformed into a binary classification problem:

```text
0 → No disease
1–4 → Disease
```

Therefore, the final project uses:

```text
0 = No disease
1 = Disease
```

---

# Features

The dataset contains the following variables:

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of the patient                    |
| `sex`      | Sex of the patient                    |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `fbs`      | Fasting blood sugar indicator         |
| `restecg`  | Resting electrocardiographic result   |
| `thalach`  | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression induced by exercise     |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia-related categorical value |
| `target`   | Binary disease target                 |

---

# Machine Learning Models

Three classification algorithms are implemented.

## 1. Logistic Regression

Logistic Regression provides a simple and interpretable baseline for binary classification.

It estimates the probability of belonging to the positive class.

---

## 2. Random Forest

Random Forest is an ensemble learning method based on multiple decision trees.

The project uses:

* `300` trees
* `class_weight="balanced"`
* fixed random state for reproducibility

Random Forest can capture nonlinear relationships between medical features.

---

## 3. Gradient Boosting

Gradient Boosting builds an ensemble of weak learners sequentially, where each new learner attempts to improve the previous model.

It can capture more complex relationships in structured data.

---

# Data Preprocessing

The preprocessing pipeline includes:

### Missing-value handling

Numerical missing values are handled using median imputation.

Categorical missing values are handled using the most frequent value.

### Feature scaling

Numerical features are standardized using `StandardScaler`.

### Train/Test Split

The dataset is divided into:

```text
80% → Training
20% → Testing
```

The split uses:

```text
random_state = 42
stratify = target
```

This preserves approximately the same class distribution in the training and testing sets.

---

# Exploratory Data Analysis

The project performs exploratory data analysis to understand the dataset before model training.

The analysis includes:

* Target class distribution
* Feature statistics
* Correlation matrix
* Age distribution
* Cholesterol distribution
* Maximum heart rate distribution
* Feature relationships with the target

Example visualizations are generated in:

```text
reports/figures/
```

---

# Model Evaluation

The models are evaluated using multiple metrics.

## Accuracy

Measures the proportion of correct predictions:

```text
Accuracy = Correct Predictions / Total Predictions
```

---

## Precision

Measures how many predicted positive cases were actually positive.

High precision means fewer false positive predictions.

---

## Recall

Measures how many actual positive cases were correctly identified.

Recall is particularly important when missing positive cases can be costly.

---

## F1 Score

The F1 score combines precision and recall into a single metric.

It is useful when the class distribution is not perfectly balanced.

---

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two classes across different classification thresholds.

The model comparison is saved in:

```text
reports/model_results.csv
```

---

# Model Explainability

A major goal of this project is to make machine learning predictions easier for humans to understand.

Instead of only returning:

```text
Prediction: 1
Probability: 82%
```

the system can provide an explanation of which features contributed most strongly to the prediction.

The explainability component is designed around the idea of:

```text
Patient Data
     ↓
Model Prediction
     ↓
Probability
     ↓
Feature Contributions
     ↓
Human-readable Explanation
```

For example, an explanation could indicate that features such as:

* chest pain type
* maximum heart rate
* ST depression
* number of major vessels
* age

contributed strongly toward or away from the model's positive prediction.

## Local Explainability

Local explainability focuses on one individual prediction.

For example:

```text
Prediction: Positive
Estimated probability: 82%

Main contributing features:
- Chest pain type → pushed prediction toward positive class
- ST depression → pushed prediction toward positive class
- Number of major vessels → pushed prediction toward positive class
- Maximum heart rate → pushed prediction away from positive class
```

The explanation describes the **model's reasoning**, not a medical diagnosis.

## Global Explainability

Global explainability examines which features are generally important across the dataset.

This can help identify the variables the model relies on most frequently.

> Explainability methods explain model behavior. They do not prove that a feature medically causes disease.

---

# Project Structure

```text
Disease-Prediction-from-Medical-Data/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── heart.csv
│   │
│   └── processed/
│
├── notebooks/
│   └── 01_disease_prediction_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   └── best_model.joblib
│
├── reports/
│   ├── model_results.csv
│   └── figures/
│       ├── logistic_regression_confusion_matrix.png
│       ├── random_forest_confusion_matrix.png
│       └── gradient_boosting_confusion_matrix.png
│
└── app/
    └── app.py
```

---

# Source Code Description

## `src/config.py`

Contains project configuration such as:

* Dataset path
* Model paths
* Report paths
* Random state
* Test size
* Target column

---

## `src/data_loader.py`

Responsible for:

* Loading the dataset
* Checking that the dataset exists
* Validating the target column
* Checking that the dataset is not empty

---

## `src/preprocessing.py`

Responsible for:

* Separating features and target
* Creating preprocessing pipelines
* Handling missing values
* Scaling numerical features
* Splitting the dataset

---

## `src/train.py`

Responsible for:

* Building the machine learning models
* Training the models
* Saving trained models

Generated models include:

```text
logistic_regression.joblib
random_forest.joblib
gradient_boosting.joblib
```

---

## `src/evaluate.py`

Responsible for:

* Loading trained models
* Generating predictions
* Calculating evaluation metrics
* Generating confusion matrices
* Comparing models
* Selecting the best model

The best model is saved as:

```text
models/best_model.joblib
```

---

## `src/predict.py`

Responsible for:

* Loading the best model
* Accepting patient data
* Generating a prediction
* Calculating the estimated probability

Example:

```text
Prediction: 1
Estimated probability: 82%
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/cheraga/Disease-Prediction-from-Medical-Data.git
```

Enter the project directory:

```bash
cd Disease-Prediction-from-Medical-Data
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## 1. Verify the Dataset

Run:

```bash
python -m src.data_loader
```

The program should confirm that the dataset was loaded successfully.

---

## 2. Train the Models

Run:

```bash
python -m src.train
```

This trains:

```text
Logistic Regression
Random Forest
Gradient Boosting
```

and saves the trained models in:

```text
models/
```

---

## 3. Evaluate the Models

Run:

```bash
python -m src.evaluate
```

This generates:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion matrices
* Model comparison results

---

## 4. Make a Prediction

Run:

```bash
python -m src.predict
```

The program returns:

```text
Prediction: ...
Estimated probability: ...%
```

---

# Example Prediction

A sample patient can be provided to the prediction module.

Example output:

```text
Prediction: 1
Estimated probability: 82.00%
```

The prediction represents the model's estimated classification according to the dataset definition.

It does **not** represent a clinical diagnosis.

---

# Results

The project compares three machine learning models:

| Model               |                 Accuracy |                Precision |                   Recall |                       F1 |                  ROC-AUC |
| ------------------- | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| Logistic Regression | Generated after training | Generated after training | Generated after training | Generated after training | Generated after training |
| Random Forest       | Generated after training | Generated after training | Generated after training | Generated after training | Generated after training |
| Gradient Boosting   | Generated after training | Generated after training | Generated after training | Generated after training | Generated after training |

The actual results are stored in:

```text
reports/model_results.csv
```

The best model is selected according to ROC-AUC.

---

# Limitations

This project has several limitations.

## Small Dataset

The dataset contains only 303 records.

Therefore, model performance can vary depending on the train/test split.

## Historical Dataset

The UCI Heart Disease dataset is a historical research dataset and may not represent modern clinical populations.

## Feature Limitations

The model only uses the features available in the dataset.

It does not incorporate:

* Medical imaging
* Laboratory panels beyond the available variables
* Electronic health records
* Longitudinal patient history
* Modern clinical measurements

## No Clinical Validation

The model has not been clinically validated.

Therefore, the reported metrics should not be interpreted as evidence of clinical effectiveness.

## Explainability Limitations

Explainability methods describe how the model arrives at a prediction.

They do not establish causal relationships between medical features and disease.

---

# Ethical and Medical Disclaimer

This project is intended **only for educational, research, and software-development purposes**.

It is **not a medical device** and should not be used to:

* Diagnose patients
* Replace physicians
* Recommend treatment
* Make medical decisions
* Determine whether someone has a disease

The predictions are generated by a machine learning model trained on a small public dataset and should not be considered medical advice.

For real-world medical applications, extensive clinical validation, appropriate regulatory approval, data-quality assessment, privacy protection, fairness evaluation, and expert medical supervision would be required.

---

# Reproducibility

The project uses a fixed random state:

```text
random_state = 42
```

This helps produce reproducible train/test splits and model results.

The preprocessing steps are implemented using Scikit-learn pipelines to reduce the risk of inconsistent transformations between training and prediction.

---

# Technologies

The project uses:

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook
* Streamlit

---

# Future Improvements

Future versions of the project can include:

### Model Improvements

* Stratified K-Fold Cross-Validation
* Hyperparameter optimization
* Grid Search
* Randomized Search
* Additional machine learning algorithms

### Explainability

* SHAP
* LIME
* Global feature importance
* Local prediction explanations

### Evaluation

* ROC curves
* Precision-Recall curves
* Calibration curves
* Confidence intervals
* Cross-validation mean ± standard deviation

### Application

* Interactive Streamlit interface
* Patient feature input form
* Probability visualization
* Human-readable prediction explanations

### Research Improvements

* Larger datasets
* External validation dataset
* Fairness analysis
* Robustness testing
* Model calibration
* Comparison with published baselines

---

# Reproducible Workflow

The complete workflow is:

```text
1. Obtain Dataset
       ↓
2. Validate Dataset
       ↓
3. Explore Data
       ↓
4. Handle Missing Values
       ↓
5. Split Features and Target
       ↓
6. Train/Test Split
       ↓
7. Preprocess Features
       ↓
8. Train Multiple Models
       ↓
9. Evaluate Models
       ↓
10. Compare Models
       ↓
11. Select Best Model
       ↓
12. Save Model
       ↓
13. Make Predictions
       ↓
14. Explain Predictions
```

---

# GitHub Repository

Project repository:

https://github.com/cheraga/Disease-Prediction-from-Medical-Data

---

# Author

**Rafik Cheraga**

Master's Student, Networks and Telecommunications Engineering

---

# License

This project is distributed under the license included in the repository.

See:

```text
LICENSE
```

for more information.

---

# Final Note

This project demonstrates a complete machine learning workflow for structured medical data, from data loading and exploratory analysis to model training, evaluation, prediction, and explainability.

The main purpose is to demonstrate **machine learning engineering, reproducibility, model evaluation, and interpretable AI**, rather than to provide a clinical diagnostic solution.
