# Customer Churn Prediction & Retention Analytics

An end-to-end machine learning project that predicts customer churn, identifies important factors associated with churn, and deploys the trained model through a Flask REST API.

## Business Problem

Customer churn can negatively impact revenue and customer lifetime value. The objective of this project is to identify customers who are likely to churn and understand the factors contributing to the model's churn predictions.

The project focuses on:

- Predicting whether a customer is likely to churn.
- Identifying important factors associated with churn.
- Comparing machine learning models using appropriate classification metrics.
- Optimizing the classification threshold for a recall-focused retention objective.
- Explaining model predictions using SHAP.
- Deploying the trained model through a Flask REST API.

## Dataset

The project uses a telecom customer churn dataset containing customer demographic information, services, contract details, payment information, tenure, and charges.

- **Records:** 7,043 customers
- **Original features:** 21
- **Target variable:** `Churn`

### Target Classes

- `Yes` — Customer churned
- `No` — Customer did not churn

## Project Workflow

```text
Business Problem
       ↓
Data Understanding
       ↓
Data Quality Check
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Data Preprocessing
       ↓
Random Forest
       ↓
XGBoost
       ↓
Hyperparameter Tuning
       ↓
Threshold Optimization
       ↓
Model Comparison
       ↓
SHAP Explainability
       ↓
Model Persistence
       ↓
Flask REST API
       ↓
API Testing
```

## Exploratory Data Analysis

The analysis examined:

- Churn distribution
- Numerical feature distributions
- Categorical feature distributions
- Churn vs. numerical features
- Churn vs. categorical features
- Feature correlations

### Key EDA Observations

Higher churn rates were observed among:

- Month-to-month contract customers
- Customers using electronic check as the payment method
- Customers without online security or technical support
- Customers with shorter tenure
- Customers with relatively higher monthly charges

These are observed associations in the dataset and should not be interpreted as causal relationships.

## Feature Engineering

Three meaningful features were created based on the EDA findings.

### TotalServices

Counts the number of subscribed services among:

- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### AvgMonthlyCharges

Calculates the average monthly charge using total charges and customer tenure.

For customers with zero tenure, the current `MonthlyCharges` value is used.

### TenureGroup

Customers are grouped into:

- New
- Early
- Established
- Long-term

## Data Preprocessing

The preprocessing workflow includes:

- Removing the `customerID` identifier.
- Stratified train-test split.
- Separating numerical and categorical features.
- One-hot encoding categorical variables.
- Fitting preprocessing only on the training data.

Tree-based models were used, so feature scaling was not required.

## Machine Learning Models

### Random Forest

A Random Forest classifier was trained as a baseline tree-based model.

### XGBoost

An XGBoost classifier was trained and then tuned using `GridSearchCV`.

The tuning process explored:

- `n_estimators`
- `learning_rate`
- `max_depth`

Recall was used as the tuning objective because identifying potential churners is important for a retention-focused use case.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Because customer churn is an imbalanced classification problem, accuracy alone was not used to evaluate the models.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 76.65% | 55.24% | 63.37% | 59.03% | 82.44% |
| XGBoost Baseline | 79.42% | 64.00% | 51.34% | 56.97% | 84.25% |
| XGBoost Tuned | 79.91% | 64.92% | 52.94% | 58.32% | 84.30% |
| XGBoost Tuned (Threshold 0.40) | 78.42% | 58.54% | 64.17% | 61.22% | 84.30% |

For the retention-focused objective, the tuned XGBoost model with a `0.40` classification threshold was selected because it provided higher recall and F1 score among the evaluated configurations.

## Threshold Optimization

The tuned XGBoost model was evaluated using a classification threshold of `0.40` instead of the default `0.50`.

```text
Probability >= 0.40 → Churn
Probability < 0.40  → No Churn
```

This increases the number of customers identified as potential churners and improves recall for the retention-focused use case.

The threshold changes the final classification decision; it does not retrain or change the underlying XGBoost model.

## SHAP Explainability

SHAP was used to understand how the trained XGBoost model makes churn predictions.

### Global Explanation

The SHAP summary plot provides a global view of feature influence across the test dataset.

Important features included:

- Contract type
- Tenure
- Internet service
- Payment method
- Monthly charges
- Total charges
- Online security

### Individual Explanation

A SHAP waterfall plot was used to explain an individual customer's prediction and show which features contributed toward or away from the model's churn prediction.

SHAP explanations describe model behavior and should not be interpreted as causal relationships.

## Model Persistence

The trained XGBoost model and preprocessing object are saved using Joblib:

```text
models/
├── churn_model.joblib
└── preprocessor.joblib
```

These saved artifacts are loaded by the Flask application to generate predictions for new customer data.

## Flask REST API

The trained model is deployed through a Flask REST API.

The Flask application is located at:

```text
app/app.py
```

The `/predict` endpoint:

1. Receives customer information as JSON.
2. Performs the required feature engineering.
3. Applies the saved preprocessing object.
4. Generates the churn probability.
5. Applies the `0.40` classification threshold.
6. Returns the prediction as JSON.

### Example API Response

```json
{
    "churn_probability": 0.5731,
    "prediction": 1,
    "prediction_label": "Churn"
}
```

The API was tested locally using Python `requests`.

## Project Structure

```text
customer-churn-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── customer_churn.csv
│
├── models/
│   ├── churn_model.joblib
│   └── preprocessor.joblib
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Flask
- Requests
- Jupyter Notebook

## Key Outcome

This project demonstrates an end-to-end machine learning workflow covering:

- Data analysis
- Data quality validation
- Feature engineering
- Machine learning model development
- Hyperparameter tuning
- Threshold optimization
- Model evaluation
- SHAP explainability
- Model persistence
- Flask REST API deployment

The final API can accept new customer information and return a churn probability and churn prediction.

## Future Improvements

Possible future improvements include:

- Adding a web-based user interface.
- Experimenting with additional classification models.
- Improving probability calibration.
- Integrating the API with a production application.
- Adding automated model monitoring and retraining workflows.
