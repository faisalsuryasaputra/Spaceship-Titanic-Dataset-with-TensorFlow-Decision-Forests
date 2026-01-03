# Spaceship-Titanic-Dataset-with-TensorFlow-Decision-Forests
This repository contains a complete and concise implementation of a baseline Random Forest model using TensorFlow Decision Forests (TF-DF) for the Spaceship Titanic Kaggle competition.
The project demonstrates an end-to-end machine learning workflow on tabular data, including exploratory data analysis (EDA), feature engineering, handling missing values, model training, evaluation, feature importance analysis, and submission generation.

# Key Highlights
Uses TensorFlow Decision Forests for robust tree-based modeling
Minimal preprocessing thanks to native TF-DF support for categorical features
Feature engineering on Cabin (Deck, Cabin_num, Side)
Model evaluation using Out-of-Bag (OOB) accuracy and validation set
Visualization of feature importance and model performance
Generates Kaggle-ready submission file

# Model
Algorithm: Random Forest Classifier
Framework: TensorFlow Decision Forests
Task: Binary Classification (Transported: True / False)

# Performance
Validation Accuracy: ~80%
Strong baseline for further hyperparameter tuning or ensemble methods

# Dataset
Spaceship Titanic Dataset (Kaggle)
Predict whether a passenger was transported to an alternate dimension
This repository is suitable as a baseline reference, learning resource, or starting point for experimentation with tree-based models on structured data.
