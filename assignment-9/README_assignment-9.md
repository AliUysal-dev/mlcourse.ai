# Assignment 9: Time Series Analysis (Prophet & ARIMA)

This directory contains the code and solution for Assignment 9 of the mlcourse.ai program, focusing on time series forecasting applied to daily Wikipedia page view counts.

## Contents

* [Jupyter Notebook Solution](https://github.com/AliUysal-dev/mlcourse.ai/blob/main/assignment-9/assignment-9.ipynb)

## Key Topics & Questions Covered

1. Preparing a daily time series (Wikipedia "Machine Learning" page views, 2015-2016) for forecasting
2. Forecasting future views with Facebook Prophet, trained on the first months of data
3. Evaluating forecast quality on a 30-day holdout using MAPE and MAE
4. Testing stationarity of the series with the Augmented Dickey-Fuller test
5. Selecting optimal SARIMAX hyperparameters `(p, d, q)(P, D, Q, 7)` via grid search, ranked by AIC

## Dataset

* Dataset: Daily view counts for the Wikipedia "Machine Learning" article (`wiki_machine_learning.csv`)
* Primary Libraries: pandas, Prophet, statsmodels
