import numpy as np
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)


# ============================================================
# Simple test data
# ============================================================

# We have 4 observations and 2 features.
# The first column is the intercept (all 1s).
tx = np.array([
    [1.0, 1.0],
    [1.0, 2.0],
    [1.0, 3.0],
    [1.0, 4.0],
])

# Target values for linear regression
y = np.array([2.0, 4.0, 6.0, 8.0])

# Initial weights
initial_w = np.zeros(tx.shape[1])


# ============================================================
# Linear regression - Gradient Descent
# ============================================================

w, loss = mean_squared_error_gd(
    y,
    tx,
    initial_w,
    max_iters=1000,
    gamma=0.01,
)

print("Linear regression - Gradient Descent")
print("Weights:", w)
print("Loss:", loss)
print()


# ============================================================
# Linear regression - Stochastic Gradient Descent
# ============================================================

w, loss = mean_squared_error_sgd(
    y,
    tx,
    initial_w,
    max_iters=1000,
    gamma=0.01,
)

print("Linear regression - Stochastic Gradient Descent")
print("Weights:", w)
print("Loss:", loss)
print()


# ============================================================
# Least Squares
# ============================================================

w, loss = least_squares(
    y,
    tx,
)

print("Least Squares")
print("Weights:", w)
print("Loss:", loss)
print()


# ============================================================
# Ridge Regression
# ============================================================

w, loss = ridge_regression(
    y,
    tx,
    lambda_=0.01,
)

print("Ridge Regression")
print("Weights:", w)
print("Loss:", loss)
print()


# ============================================================
# Logistic regression data
# ============================================================

# Same feature matrix, but now the targets must be 0 or 1.
y_logistic = np.array([0.0, 0.0, 1.0, 1.0])

# Initial weights for logistic regression
initial_w_logistic = np.zeros(tx.shape[1])


# ============================================================
# Logistic Regression
# ============================================================

w, loss = logistic_regression(
    y_logistic,
    tx,
    initial_w_logistic,
    max_iters=1000,
    gamma=0.1,
)

print("Logistic Regression")
print("Weights:", w)
print("Loss:", loss)
print()


# ============================================================
# Regularized Logistic Regression
# ============================================================

w, loss = reg_logistic_regression(
    y_logistic,
    tx,
    lambda_=0.01,
    initial_w=initial_w_logistic,
    max_iters=1000,
    gamma=0.1,
)

print("Regularized Logistic Regression")
print("Weights:", w)
print("Loss:", loss)
print()
