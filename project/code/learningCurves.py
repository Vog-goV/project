import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = np.load('../MovieSet_MODIFIED.npz')

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


# ===========================
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples\n")

poly = PolynomialFeatures(degree=2, include_bias=False) # [2 degree] is maximum usable for 46 features and [7 degree] is for 7 features(more leads to overfitting)
Xp_train = poly.fit_transform(X_train)
Xp_test = poly.transform(X_test)


train_mses_unreg = []
test_mses_unreg = []

train_mses_ridge = []
test_mses_ridge = []

train_mses_lasso = []
test_mses_lasso = []

m_total = len(Xp_train)
sizes = [int(m_total * frac) for frac in np.linspace(0.01, 1.0, 15)]

for size in sizes:
    X_subset = Xp_train[:size]
    y_subset = y_train[:size]
    
    # ==== Unregularized ====
    model_unreg = LinearRegression()
    model_unreg.fit(X_subset, y_subset)
    train_mse_unreg = mean_squared_error(y_subset, model_unreg.predict(X_subset))
    train_mses_unreg.append(train_mse_unreg)
    test_mse_unreg = mean_squared_error(y_test, model_unreg.predict(Xp_test))
    test_mses_unreg.append(test_mse_unreg)
    # ===============

    # ==== Ridge ====
    model_ridge = Ridge(alpha=1.0, random_state=42)
    model_ridge.fit(X_subset, y_subset)
    train_mse_ridge = mean_squared_error(y_subset, model_ridge.predict(X_subset))
    train_mses_ridge.append(train_mse_ridge)
    test_mse_ridge = mean_squared_error(y_test, model_ridge.predict(Xp_test))
    test_mses_ridge.append(test_mse_ridge)
    # ===============

    # ==== Lasso ====
    lasso_model = Lasso(alpha=0.001, random_state=42, max_iter=4_000)
    lasso_model.fit(X_subset, y_subset)
    train_mse_lasso = mean_squared_error(y_subset, lasso_model.predict(X_subset))
    train_mses_lasso.append(train_mse_lasso)
    test_mse_lasso = mean_squared_error(y_test, lasso_model.predict(Xp_test))
    test_mses_lasso.append(test_mse_lasso)
    # ===============

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].plot(sizes, train_mses_unreg, label='Train MSE (Unreg)', marker='o', color='blue', linewidth=2)
ax[0].plot(sizes, test_mses_unreg, label='Test MSE (Unreg)', marker='o', color='red', linewidth=2)
ax[0].set_xlabel('Number of train set')
ax[0].set_ylabel('Mean Squared Error (MSE)')
ax[0].set_title('Learning Curves (Unreg)')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(sizes, train_mses_ridge, label='Train MSE (Ridge)', marker='o', color='blue', linewidth=2)
ax[1].plot(sizes, test_mses_ridge, label='Test MSE (Ridge)', marker='o', color='red', linewidth=2)
ax[1].set_xlabel('Number of train set')
ax[1].set_ylabel('Mean Squared Error (MSE)')
ax[1].set_title('Learning Curves (Ridge)')
ax[1].legend()
ax[1].grid(True)

ax[2].plot(sizes, train_mses_lasso, label='Train MSE (Lasso)', marker='o', color='blue', linewidth=2)
ax[2].plot(sizes, test_mses_lasso, label='Test MSE (Lasso)', marker='o', color='red', linewidth=2)
ax[2].set_xlabel('Number of train set')
ax[2].set_ylabel('Mean Squared Error (MSE)')
ax[2].set_title('Learning Curves (Lasso)')
ax[2].legend()
ax[2].grid(True)


plt.tight_layout()
plt.savefig('../EDA/LearningCurves.png')
plt.show()