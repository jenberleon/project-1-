import numpy as np


def compute_mse(y, tx, w):
    """Compute mean squared error with a factor 0.5."""
    error = y - tx @ w
    return 0.5 * np.mean(error ** 2)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Parameters
    ----------
    y : np.ndarray
        Target values, shape (N,).
    tx : np.ndarray
        Feature matrix, shape (N, D).
    initial_w : np.ndarray
        Initial weight vector, shape (D,).
    max_iters : int
        Number of gradient descent iterations.
    gamma : float
        Step size.

    Returns
    -------
    w : np.ndarray
        Final weight vector.
    loss : float
        MSE loss at the final weight vector.
    """

    w = initial_w.copy()

    for _ in range(max_iters):

        # Compute prediction error
        error = y - tx @ w

        # Compute gradient
        gradient = -(tx.T @ error) / len(y)

        # Update weights
        w = w - gamma * gradient

    # Compute loss using final w
    loss = compute_mse(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    Mini-batch size = 1.
    """

    w = initial_w.copy()

    for _ in range(max_iters):

        # Randomly select one data point
        i = np.random.randint(len(y))

        # Select one sample
        x_i = tx[i]
        y_i = y[i]

        # Prediction error for this sample
        error = y_i - x_i @ w

        # Gradient for this sample
        gradient = -x_i * error

        # Update weights
        w = w - gamma * gradient

    # Compute loss on the complete dataset
    loss = compute_mse(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using the normal equations.

    Uses np.linalg.solve(), which is allowed.
    """

    # Compute X^T X
    A = tx.T @ tx

    # Compute X^T y
    b = tx.T @ y

    # Solve:
    # (X^T X) w = X^T y
    w = np.linalg.solve(A, b)

    # Compute loss
    loss = compute_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using the normal equations.

    The regularization term is used to calculate w,
    but is NOT included in the returned loss.
    """

    # Number of samples
    n = len(y)

    # Number of features
    d = tx.shape[1]

    # Compute:
    # X^T X + N * lambda * I
    A = tx.T @ tx + 2*n * lambda_ * np.eye(d)

    # Compute X^T y
    b = tx.T @ y

    # Solve:
    # (X^T X + N lambda I) w = X^T y
    w = np.linalg.solve(A, b)

    # IMPORTANT:
    # The returned loss does NOT contain the ridge penalty.
    loss = compute_mse(y, tx, w)

    return w, loss


def sigmoid(t):
    """
    Compute the sigmoid function.

    sigmoid(t) = 1 / (1 + exp(-t))
    """

    return 1.0 / (1.0 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """
    Compute logistic regression loss.

    This loss does not contain a regularization penalty.
    """

    # Compute predicted probabilities
    pred = sigmoid(tx @ w)

    # Avoid log(0)
    eps = 1e-15
    pred = np.clip(pred, eps, 1 - eps)

    # Binary cross-entropy loss
    loss = -np.mean(
        y * np.log(pred)
        + (1 - y) * np.log(1 - pred)
    )

    return loss


def compute_logistic_gradient(y, tx, w):
    """
    Compute the gradient of the logistic loss.
    """

    # Predicted probabilities
    pred = sigmoid(tx @ w)

    # Gradient
    gradient = tx.T @ (pred - y) / len(y)

    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    y must contain values in {0, 1}.
    """

    w = initial_w.copy()

    for _ in range(max_iters):

        # Compute gradient
        gradient = compute_logistic_gradient(y, tx, w)

        # Update weights
        w = w - gamma * gradient

    # Compute final loss
    loss = compute_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma
):
    """
    Regularized logistic regression using gradient descent.

    The regularization term is used during optimization,
    but is NOT included in the returned loss.

    y must contain values in {0, 1}.
    """

    w = initial_w.copy()

    for _ in range(max_iters):

        # Gradient of ordinary logistic regression
        gradient = compute_logistic_gradient(y, tx, w)

        # Add regularization gradient
        gradient += lambda_ * w

        # Update weights
        w = w - gamma * gradient

    # IMPORTANT:
    # Do not include the regularization penalty here.
    loss = compute_logistic_loss(y, tx, w)

    return w, loss
