# Conformal Quantile HPO Code

## Overview

This repository provides reproducible code for all novel and existing conformal quantile HPO methods implemented in *"Enhancing Adaptiveness and Sampling Performance in Conformal Hyperparameter Optimization"*.

This version removes or obfuscates all identifiers, author(s) information, commit history, documentation, CI/CD that could lead to identifiable information.

If the submission is successful, the paper will be updated to point to a repository restoring all stripped or obfuscated information.

## General Usage

The example below shows how to optimize hyperparameters for a RandomForest classifier.

### Step 1: Import Required Libraries

```python
from ccqr_optimization.tuning import ConformalTuner
from ccqr_optimization.wrapping import IntRange, FloatRange, CategoricalRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
We import the necessary libraries for tuning and model evaluation. The `load_wine` function is used to load the wine dataset, which serves as our example data for optimizing the hyperparameters of the RandomForest classifier.

### Step 2: Define the Objective Function

```python
def objective_function(configuration):
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=configuration['n_estimators'],
        max_features=configuration['max_features'],
        criterion=configuration['criterion'],
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return accuracy_score(y_test, predictions)
```
This function defines the objective we want to optimize. It loads the wine dataset, splits it into training and testing sets, and trains a RandomForest model using the provided configuration. The function returns the accuracy score, which serves as the optimization metric.

### Step 3: Define the Search Space

```python
search_space = {
    'n_estimators': IntRange(50, 200),
    'max_features': FloatRange(0.1, 1.0),
    'criterion': CategoricalRange(['gini', 'entropy', 'log_loss'])
}
```
Here, we specify the search space for hyperparameters. This includes defining the range for the number of estimators, the proportion of features to consider when looking for the best split, and the criterion for measuring the quality of a split.

### Step 4: Create and Run the Tuner

```python
tuner = ConformalTuner(
    objective_function=objective_function,
    search_space=search_space,
    minimize=False
)
tuner.tune(max_searches=50, n_random_searches=10)
```
We initialize the `ConformalTuner` with the objective function and search space. The tuner is then run to find the best hyperparameters by maximizing the accuracy score.

### Step 5: Retrieve and Display Results

```python
best_params = tuner.get_best_params()
best_score = tuner.get_best_value()

print(f"Best accuracy: {best_score:.4f}")
print(f"Best parameters: {best_params}")
```
Finally, we retrieve the best parameters and score from the tuning process and print them to the console for review.
