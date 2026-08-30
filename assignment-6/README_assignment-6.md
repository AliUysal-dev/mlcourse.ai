# Assignment 6: Catch Me If You Can - Intruder Detection (Alice)

This directory contains the code and solution for Assignment 6 of the mlcourse.ai program, my first Kaggle competition: a session-based user identification task on the "Catch Me If You Can" (Alice) dataset.

## Contents

* [Jupyter Notebook Solution](https://github.com/AliUysal-dev/mlcourse.ai/blob/main/assignment-6/assignment-6.ipynb)

## Key Topics & Questions Covered

1. Exploratory analysis of session-based browsing data (`site1..site10`, `time1..time10`)
2. Investigating severe class imbalance (~0.9% positive class) and choosing ROC-AUC as the evaluation metric
3. Feature engineering: encoding thousands of unique site IDs as sparse bag-of-words vectors with `CountVectorizer`
4. Extracting temporal features (`hour`, `weekday`) from session timestamps and identifying behavioral patterns
5. Avoiding data leakage: fitting `StandardScaler` only on the training set
6. Combining sparse site features with scaled numerical features using `scipy.sparse.hstack`
7. Training a Logistic Regression classifier and validating with a stratified train/validation split
8. Submitting predictions to Kaggle via the `kaggle` CLI

## Dataset

* Dataset: Session logs from a shared computer, labeled by whether the session belongs to "Alice" or another user
* Files: `train_sessions.csv`, `test_sessions.csv`, `site_dic.pkl`
* Primary Libraries: pandas, scikit-learn, scipy

## Result

* Public leaderboard score: **0.91934** (ROC-AUC)
* Private leaderboard score: **0.93048** (ROC-AUC)
* Model: Logistic Regression on site bag-of-words + hour/weekday features
