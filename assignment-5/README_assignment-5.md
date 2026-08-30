# Assignment 5: Logistic Regression and Random Forest in the Credit Scoring Problem

This directory contains the code and solution for Assignment 5 of the mlcourse.ai program, applying classification models to predict serious loan delinquency.

## Contents

* [Jupyter Notebook Solution](https://github.com/AliUysal-dev/mlcourse.ai/blob/main/assignment-5/assignment-5.ipynb)

## Key Topics & Questions Covered

1. Exploring the credit scoring dataset and the target distribution (`SeriousDlqin2yrs`)
2. Handling missing values by filling with the median of each column
3. Tuning a `LogisticRegression` model (with `class_weight="balanced"`) over a grid of `C` values using `StratifiedKFold` cross-validation
4. Identifying the most influential feature for predicting delinquency
5. Training a `BaggingClassifier` of logistic regressors and tuning `max_features`, `max_samples`, and base estimator `C` via `RandomizedSearchCV`
6. Comparing single-model interpretability against ensemble performance and correlation between base estimators

## Dataset

* Dataset: Credit scoring sample, borrower financial and demographic features (`credit_scoring_sample.csv`)
* Target: `SeriousDlqin2yrs` — whether the borrower experienced serious delinquency (~22.2% positive class)
* Primary Libraries: pandas, scikit-learn

## Result (final answers)

* Best regularization strength for tuned Logistic Regression: **C = 0.001**
* Most important feature: `NumberOfTime30-59DaysPastDueNotWorse`
* Least important feature (for the ensemble): `NumberOfDependents`
* Tuned Bagging ensemble accuracy: **~80.75%**
* Ensembling improves performance mainly due to lower correlation between individual base models
