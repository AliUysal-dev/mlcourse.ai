# Assignment 8: Implementation of an Online (SGD) Regressor

This directory contains the code and solution for Assignment 8 of the mlcourse.ai program, implementing Stochastic Gradient Descent linear regression from scratch and comparing it to scikit-learn's implementation.

## Contents

* [Jupyter Notebook Solution](https://github.com/AliUysal-dev/mlcourse.ai/blob/main/assignment-8/assignment-8.ipynb)

## Key Topics & Questions Covered

1. Implementing a custom `SGDRegressor` class (`BaseEstimator`) with per-sample weight updates
2. Preparing and scaling the weight/height dataset for a simple regression task
3. Training the custom SGD regressor and tracking MSE across training iterations
4. Visualizing the training process: MSE curve and the evolution of weights (`w0`, `w1`) over iterations
5. Evaluating the custom model's MSE on a hold-out validation set
6. Training scikit-learn's `LinearRegression` on the same data and comparing hold-out MSE against the custom SGD implementation

## Dataset

* Dataset: Weights and heights (`weights_heights.csv`), a simple single-feature regression example
* Primary Libraries: numpy, pandas, scikit-learn, matplotlib

## Result

* Custom SGD regressor: minimal training MSE **2.7187**, hold-out MSE **2.6759**
* Scikit-learn `LinearRegression`: hold-out MSE **2.6708**
* The two hold-out MSE values are very close but not identical within the assignment's strict tolerance, indicating the custom SGD implementation converges near — but not exactly to — the closed-form OLS solution
