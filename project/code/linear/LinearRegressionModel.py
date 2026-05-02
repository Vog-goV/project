import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    LinearRegression,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === Implementation with Numpy ===
class LinearRegressionBatchGD:
    """
    Linear regression trained with batch gradient descent on MSE.
    If add_intercept=True, prepends a column of ones for the bias term.
    """

    def __init__(self, learning_rate=0.05, n_iterations=3000, add_intercept=True, tol=1e-10):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.add_intercept = add_intercept
        self.tol = tol
        self.theta = None
        self.train_loss_history = []
        self.test_loss_history = []

    def _design_matrix(self, X):
        X = np.asarray(X, dtype=float)
        if self.add_intercept:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def fit(self, X, y, X_test=None, y_test=None):
        Xd = self._design_matrix(X)
        y = np.asarray(y, dtype=float).ravel()

        if X_test is not None and y_test is not None:
            Xd_test = self._design_matrix(X_test)
            y_test = np.asarray(y_test, dtype=float).ravel()

        m, n = Xd.shape
        self.theta = np.zeros(n)

        self.train_loss_history = []
        self.test_loss_history = []

        for it in range(self.n_iterations):
            preds = Xd.dot(self.theta)
            errors = preds - y
            # Loss = (1/2m) * sum(errors^2); grad = (1/m) * Xd.T @ errors
            self.train_loss_history.append(np.mean(errors ** 2))

            if X_test is not None and y_test is not None:
                test_preds = Xd_test.dot(self.theta)
                test_errors = test_preds - y_test
                self.test_loss_history.append(np.mean(test_errors ** 2))

            grad = (1.0 / m) * Xd.T.dot(errors)
            self.theta -= self.learning_rate * grad
            if it > 0 and abs(self.train_loss_history[-1] - self.train_loss_history[-2]) < self.tol:
                break
        return self

    def predict(self, X):
        return self._design_matrix(X).dot(self.theta)

    @property
    def intercept_(self):
        return float(self.theta[0]) if self.add_intercept else 0.0

    @property
    def coef_(self):
        return self.theta[1:].copy() if self.add_intercept else self.theta.copy()


def mse_manual(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# data_df = pd.read_csv("MovieSet.csv")
# target_col = 'rating'
# feature_cols = [c for c in data_df.columns if c != target_col]
# print(f"\nData Shape: {data_df.shape}")
# print(f"Features: {feature_cols}")
# print(data_df.head(), '\n')

# data = np.load('MovieSet.npz')
# X = data['X']
# y = data['y']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


data_df = pd.read_csv("../../MovieSet_MODIFIED.csv")
target_col = 'rating'
feature_cols = [c for c in data_df.columns if c != target_col]
print(f"\nData Shape: {data_df.shape}")
print(f"Features: {feature_cols}")
print(data_df.head(), '\n')

data = np.load('../../MovieSet_MODIFIED.npz')

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']



# ===========================
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples\n")


model_numpy = LinearRegressionBatchGD(learning_rate=0.05, n_iterations=4000, add_intercept=False)
model_numpy.fit(X_train, y_train, X_test=X_test, y_test=y_test) # X_test|y_test is for loss recording 

y_pred_train_numpy = model_numpy.predict(X_train)
y_pred_test_numpy = model_numpy.predict(X_test)
print(f'Iterations: {len(model_numpy.train_loss_history)}\n')


mse_np_train = mse_manual(y_train, y_pred_train_numpy)
r2_np_train = r2_manual(y_train, y_pred_train_numpy)
mse_np_test = mse_manual(y_test, y_pred_test_numpy)
r2_np_test = r2_manual(y_test, y_pred_test_numpy)

print(f"MSE(Numpy) on (Train) {mse_np_train:.4f}")
print(f"MSE(Numpy) on (Test) {mse_np_test:.4f}")
print("================================")
print(f"R2(Numpy) on (Train) {r2_np_train:.4f}")
print(f"R2(Numpy) on (Test) {r2_np_test:.4f}\n\n")
# === ===================== ===


# === Implementation with Scikit-Learn ===
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)

y_pred_sklearn_test = model_sklearn.predict(X_test)
y_pred_sklearn_train = model_sklearn.predict(X_train)

mse_sklearn_test = mean_squared_error(y_test, y_pred_sklearn_test)
r2_sklearn_test = r2_score(y_test, y_pred_sklearn_test)
mse_sklearn_train = mean_squared_error(y_train, y_pred_sklearn_train)
r2_sklearn_train = r2_score(y_train, y_pred_sklearn_train)

print(f"MSE(Sklearn) on (Train) {mse_sklearn_train:.4f}")
print(f"MSE(Sklearn) on (Test) {mse_sklearn_test:.4f}")
print("================================")
print(f"R2(Sklearn) on (Train) {r2_sklearn_train:.4f}")
print(f"R2(Sklearn) on (Test) {r2_sklearn_test:.4f}\n\n")
# === ================================ ===





print(y_test[1116])
print(model_numpy.predict(X_test[1116].reshape(1, -1)))