# Assignment 3: Decision Trees with a Toy Task and the UCI Adult Dataset

This directory contains the code and solution for Assignment 3 of the mlcourse.ai program, covering the theory of entropy and information gain, and applying decision trees to a real classification problem.

## Contents

* [Jupyter Notebook Solution](https://github.com/AliUysal-dev/mlcourse.ai/blob/main/assignment-3/assignment-3.ipynb)

## Key Topics & Questions Covered

1. Building a toy decision tree by hand on a small categorical dataset (looks, alcohol, eloquence, spending)
2. Calculating Shannon entropy of the initial dataset and of a candidate split
3. Computing information gain (IG) for a split on the "Looks" feature
4. Implementing `entropy()` and `information_gain()` from scratch and validating with warm-up examples (colored balls, a fair die)
5. Training a `DecisionTreeClassifier` on the toy dataset and visualizing it with `plot_tree`
6. Applying the same techniques to the UCI Adult Census Income dataset: one-hot encoding categorical features, aligning train/test columns
7. Comparing an untuned decision tree (`max_depth=3`) against one tuned via `GridSearchCV` over `max_depth`

## Dataset

* Toy dataset: a small hand-crafted categorical dataset (course example)
* Real dataset: UCI Adult Census Income (`adult_train.csv`, `adult_test.csv`)
* Primary Libraries: pandas, scikit-learn, pydotplus

## Result

* Initial system entropy (S0): **0.9852**
* Untuned decision tree (`max_depth=3`) accuracy on Adult test set: **0.8361**
* Tuned decision tree (best `max_depth` via GridSearchCV) accuracy: **0.8087**
