# Assignment 4: Exploring OLS, Lasso and Random Forest in a Regression Task

This directory contains the code and solution for Assignment 4 of the mlcourse.ai program, comparing linear and ensemble regression models on a wine quality prediction task.

## Contents

* [Jupyter Notebook Solution](https://github.com/AliUysal-dev/mlcourse.ai/blob/main/assignment-4/assignment-4.ipynb)

## Key Topics & Questions Covered

1. Preparing a 7:3 train/holdout split and scaling features with `StandardScaler`
2. Training an Ordinary Least Squares linear regression model and measuring MSE on train and holdout sets
3. Identifying the most influential feature according to the linear model's coefficients
4. Training a Lasso model (weak regularization, `alpha=0.01`) and finding the least informative feature
5. Tuning Lasso's regularization strength with `LassoCV` over a log-spaced grid of alphas via 5-fold cross-validation
6. Training a default Random Forest regressor and evaluating MSE on train, cross-validation, and holdout sets
7. Tuning `max_depth` and `max_features` with `GridSearchCV` and re-evaluating performance
8. Extracting and ranking Random Forest feature importances

## Dataset

* Dataset: Wine Quality (white wine), physicochemical properties predicting quality score (`winequality-white.csv`)
* Primary Libraries: pandas, scikit-learn

## Result

* Linear Regression: Train MSE **0.5581**, Holdout MSE **0.5842**; most influential feature: `density`
* Lasso (alpha=0.01): least informative feature: `fixed acidity`
* LassoCV (best alpha ≈ 0.00028): Train MSE **0.5581**, Holdout MSE **0.5833**; least informative feature: `citric acid`
* Random Forest (default): Train MSE **0.0526**, CV MSE **0.4142**, Holdout MSE **0.3716**
* Random Forest (tuned, `max_depth=20`, `max_features='sqrt'`): CV MSE **0.3993**, Holdout MSE **0.3689**
* Most important feature (Random Forest): `alcohol`
