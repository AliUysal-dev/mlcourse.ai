# Assignment 10: Gradient Boosting - Flight Delays (Kaggle)

This directory contains the code and solution for Assignment 10 of the mlcourse.ai program, a Kaggle Inclass competition predicting whether a flight departure will be delayed by more than 15 minutes.

## Contents

* [Jupyter Notebook Solution](https://github.com/AliUysal-dev/mlcourse.ai/blob/main/assignment-10/assignment-10.ipynb)

## Key Topics & Questions Covered

1. Working with categorical flight data (`Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest`, `Distance`, `DepTime`)
2. Feature engineering: deriving a `Flight` route feature from `Origin` and `Dest`
3. One-hot encoding high-cardinality categorical features with `LabelBinarizer`, combined into a sparse matrix with `scipy.sparse.hstack`
4. Training and comparing Logistic Regression and XGBoost classifiers
5. Blending model predictions with a weighted average (`w1 * p_logit + (1-w1) * p_xgb`) and tuning the blend weight
6. Submitting predictions to Kaggle via the `kaggle` CLI

## Dataset

* Dataset: U.S. domestic flight records with departure delay labels (`flight_delays_train.csv`, `flight_delays_test.csv`)
* Primary Libraries: pandas, scikit-learn, xgboost, scipy

## Result

* Public leaderboard score: **0.71755** (ROC-AUC)
* Private leaderboard score: **0.71755** (ROC-AUC)
* Model: Blend of Logistic Regression and XGBoost on OHE categorical + numerical features (beat 1st benchmark of 0.68202)
