# Credit Risk Analysis - Supervised Machine Learning

## Challenge - Overview

In this look at Supervised Machine Learning we use Oversampling and Undersampling techniques and various modelling methods to assess credit card risk. In the real world this is an unbalanced classification problem, so we need to look at the best way to get fair samples to train our models and compare the results. The dataset contained our target variable `loan_status` which splits the data into `high_risk` and `low_risk`. The goal of the machine learning algorithms is to predict, based on similar data provided, which class an application falls into.

For oversampling we use the `RandomOverSampler` and `SMOTE` algorithms, and for undersampling we use `ClusterCentroids`. Combining these methods, we also make use of `SMOTEENN`. Using the newly resampled data we use `LogisticRegression` modelling to compare them against each other, by producing a `balanced_accuracy_score`, a `confusion_matrix`, and a `classification_report`. We explored the machine learning models `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` after this, passing in split data and comparing them in the same way as the sampling examples.

<!--
## Challenge - Results


Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models (15 pt)


## Challenge - Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

There is a summary of the results (2 pt)
There is a recommendation on which model to use, or there is no recommendation with a justification (3 pt)
-->

## Context

This is the Challenge Repo for Module 17 of the University of Toronto School of Continuing Studies Data Analysis Bootcamp Course - **Supervised Machine Learning** - Python, Machine Learning, Scikit-Learn, Regression, and Classification. Following the guidance of the module we end up pushing this selection of files to GitHub.
