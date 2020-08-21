# Predict Category

## Problem description
You have recently launched your own app-based start-up where people can raise requests to get any kind of their problems solved related to their house or cars. Now based on certain features you have to predict what category of problems are coming up the most, whether its related to personally owned vehicles or property or anything else. Class 1 represent problems related to their own houses, 2 for personally owned vehicles and 0 refers to any other task. You want to build a predictive model so as to classify the category of problem and you want to analyse what kind of problems might come up in the future so as to modify you hiring and other procedures accordingly.

## Approach
Though this data was time series I went with solving it first without time feature. I didn't get enough time to implement below two approaches:
1. Treating the cleaned data feature as discrete (this is always the fastest approach).
2. Treating time as continous, this is the most accurate method but works for only a few datasets.
3. I also treid xgboost which didn't work well on this dataset.

## Steps
1. Removing non related features
2. Creating preprocessing pipeline
3. Training the model (the best for me was Adaboost - Logistic Regression estimators)
4. Putting output in submission file

## Learnings
1. Creating ML Pipelines
2. Adaptive boosting estimators
3. Perfomance optimization of cascaded estimators

## Improvements
1. Treating time as continous feature and using Time based algorithms
2. Doing extensive exploratory data analysis
3. Performing dimesnionality reduction - i skipped this as the data has manageable dimensions
