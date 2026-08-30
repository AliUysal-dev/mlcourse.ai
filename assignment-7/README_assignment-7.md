# Assignment 7: Unsupervised Learning (PCA & Clustering)

This directory contains the code and solution for Assignment 7 of the mlcourse.ai program, exploring dimensionality reduction and clustering on human activity recognition sensor data.

## Contents

* [Jupyter Notebook Solution](https://github.com/AliUysal-dev/mlcourse.ai/blob/main/assignment-7/assignment-7.ipynb)

## Key Topics & Questions Covered

1. Combining the UCI HAR train/test sets and scaling all 561 sensor features
2. Applying PCA to find the minimum number of components explaining 90% of variance
3. Visualizing the data projected onto the first two principal components
4. Clustering the PCA-reduced data with K-Means (`n_clusters=6`) and comparing cluster assignments to true activity labels
5. Finding the optimal number of clusters via the elbow method
6. Applying Agglomerative (hierarchical) clustering and comparing it to K-Means using the Adjusted Rand Index (ARI)
7. Training a `LinearSVC` classifier with `GridSearchCV`-tuned `C`, and evaluating precision/recall per activity class
8. Comparing classification accuracy using all 561 original features versus PCA-reduced features

## Dataset

* Dataset: UCI Human Activity Recognition (HAR) — smartphone accelerometer/gyroscope readings labeled with 6 activities (walking, walking upstairs/downstairs, sitting, standing, laying)
* Primary Libraries: pandas, scikit-learn, seaborn, matplotlib

## Result

* Minimum principal components to cover 90% variance: **65**
* Variance explained by the first principal component: **~51%**
* K-Means Adjusted Rand Index: **0.4198**
* Agglomerative Clustering Adjusted Rand Index: **0.4936** (better than K-Means)
* Best `C` for LinearSVC (cross-validation): **0.1**, with CV accuracy **~93.8%**
* SVM test set accuracy: **96.2%**; activity with weakest precision/recall: **sitting/standing (classes 4 and 5)**
