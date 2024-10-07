# Project 1 

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

## 1. Overview

This project implements an **Elastic Net regression model** (ElasticNetModel_v2) that combines L1 (Lasso) and L2 (Ridge) regularization techniques. It is designed to handle high-dimensional datasets, feature selection, and multicollinearity problems. The model uses **gradient descent** for optimization, making it suitable for large datasets.

## 2. Class and Function Descriptions

### `ElasticNetModel`

The `ElasticNetModel` class implements the Elastic Net regression model. Below are the key methods and their descriptions:

#### `__init__(self, lambdas=0.1, thresh=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01)`

Initializes the Elastic Net model with the following parameters:

- **`lambdas`** (*float*, default=0.1): Penalty coefficient for regularization.
- **`thresh`** (*float*, default=0.5): Mixing parameter between L1 and L2 regularization. Value ranges from 0 to 1, where 0 means L1 regularization only and 1 means L2 regularization only.
- **`max_iter`** (*int*, default=1000): Maximum number of iterations for gradient descent.
- **`tol`** (*float*, default=1e-4): Tolerance for the stopping condition. If changes in weights are smaller than `tol`, training stops.
- **`learning_rate`** (*float*, default=0.01): Step size for gradient descent updates.

#### `fit(self, X, y)`

Trains the Elastic Net regression model using gradient descent on the provided feature matrix `X` and target values `y`.

- **Parameters**:
  - **`X`** (*numpy array*): Feature matrix of the training dataset (without intercept).
  - **`y`** (*numpy array*): Target values corresponding to the training dataset.
  
- **Returns**:
  - An instance of `ElasticNetResults`, containing the trained weight coefficients and intercept.

#### `ElasticNetResults`

This class stores the results after training the Elastic Net model.

##### `predict(self, X)`

Predicts target values for the provided feature matrix `X`.

- **Parameters**:
  - **`X`** (*numpy array*): Feature matrix of the test dataset.
  
- **Returns**:
  - Predicted target values.

### Example Usage

```python
from elasticnet import ElasticNetModel

# Initialize the model
model = ElasticNetModel(lambdas=0.1, thresh=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01)

# Fit the model on training data
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)
```
