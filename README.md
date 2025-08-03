# Orange_Telco_Churn

 About the Company
Orange is a leading telecommunications provider offering mobile, internet, and landline services. Like most telecom companies, Orange faces a high customer churn rate, impacting revenue and growth. This project addresses churn by predicting which customers are likely to leave and providing actionable retention strategies.

# Problem Statement (Need for the Project)
Churn directly impacts revenue and market share.

It is cheaper to retain customers than to acquire new ones.

Predicting churn allows proactive retention campaigns, reducing losses.

# Dataset
Source: Orange Telco Churn Dataset (classification problem)

Features: Customer demographics, account details, service usage, billing, and plan details

Target Variable: Churn (Yes/No)

# Objectives
Predict whether a customer will churn.

Identify key factors influencing churn.

Provide actionable recommendations to reduce churn.

# Plan to Solve the Problem (PACE Model)
P — Plan
Understand the churn business case.

Collect & preprocess Orange dataset.

Choose ML algorithms suitable for imbalanced classification.

A — Analyze
Performed EDA to identify key churn drivers (contract type, tenure, charges, etc.).

Detected class imbalance in churn variable.

C — Construct
Applied preprocessing (encoding, missing values handling).

Used class_weight='balanced' to address imbalance.

Trained multiple models (Logistic Regression, Random Forest, XGBoost).

Selected Voting Classifier (Ensemble) as final model.

E — Execute
Pickled the trained model for deployment.

Created a Flask web application to take user input and predict churn.

Integrated predefined churn reasons & suggestions for explainability.

Deployed on Render.

# Model Used & Performance
We tested multiple models:

Logistic Regression

Decision Trees

Random Forest

Gradient Boosting

Voting Classifier (Final Model)

✅ Final Accuracy: 96%
✅ Balanced Precision & Recall to avoid bias towards non-churn customers



# Challenges & Solutions
Challenge	Solution
Imbalanced churn dataset	Used class_weight='balanced' in the model to handle class imbalance
Overfitting in complex models	Applied cross-validation and hyperparameter tuning to regularize models
Model explainability for business users	Provided predefined churn reasons and suggestions based on feature importance and domain understanding
Deployment dependency issues	Specified Python runtime and cleaned requirements.txt for Render deployment

# Deployment
Platform: Render

Tech Stack: Flask + HTML/CSS + Python

Accessible via: https://orange-telcom-churn.onrender.com

# STAR Method Summary
Situation: Orange faced increasing customer churn impacting revenue.

Task: Predict churn and suggest actionable interventions.

Action: Built an ML pipeline with balanced class weights, tuned ensemble model, predefined churn reasons, and Flask deployment.

Result: Achieved ~82% accuracy, deployed a live predictive system enabling proactive retention strategies.
