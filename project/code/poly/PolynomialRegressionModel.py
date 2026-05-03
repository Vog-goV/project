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


# data_df = pd.read_csv("../../MovieSet.csv")
# target_col = 'rating'
# feature_cols = [c for c in data_df.columns if c != target_col]
# print(f"\nData Shape: {data_df.shape}")
# print(f"Features: {feature_cols}")
# print(data_df.head(), '\n')

# data = np.load('../../MovieSet.npz')
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
# print(data_df.head(), '\n')

data = np.load('../../MovieSet_MODIFIED.npz')

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


# ===========================
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples\n")

poly = PolynomialFeatures(degree=2, include_bias=False)
Xp_train = poly.fit_transform(X_train)
Xp_test = poly.transform(X_test)

print(f"\nNumber of features before: {X_train.shape[1]}")
print(f"Number of features on unregularized model: {Xp_train.shape[1]}\n")

# ==== Unregularized ====
unreg_model = LinearRegression()
unreg_model.fit(Xp_train, y_train)

unreg_model_mse_train = mean_squared_error(y_train, unreg_model.predict(Xp_train))
unreg_model_mse_test = mean_squared_error(y_test, unreg_model.predict(Xp_test))
unreg_model_r2_train = r2_score(y_train, unreg_model.predict(Xp_train))
unreg_model_r2_test = r2_score(y_test, unreg_model.predict(Xp_test))

print("==== Unregularized ====")
print(f"MSE(Unregularized) on (Train) {unreg_model_mse_train:.6f}")
print(f"R2(Unregularized) on (Train) {unreg_model_r2_train:.6f}\n--")
print(f"MSE(Unregularized) on (Test) {unreg_model_mse_test:.6f}")
print(f"R2(Unregularized) on (Test) {unreg_model_r2_test:.6f}\n")


# ==== Ridge ====
ridge_model = Ridge(alpha=1.0, solver='svd')
ridge_model.fit(Xp_train, y_train)

ridge_model_mse_train = mean_squared_error(y_train, ridge_model.predict(Xp_train))
ridge_model_mse_test = mean_squared_error(y_test, ridge_model.predict(Xp_test))
ridge_model_r2_train = r2_score(y_train, ridge_model.predict(Xp_train))
ridge_model_r2_test = r2_score(y_test, ridge_model.predict(Xp_test))

print("==== Ridge (L2) ====")
print(f"MSE(Ridge (L2)) on (Train) {ridge_model_mse_train:.6f}")
print(f"R2(Ridge (L2)) on (Train) {ridge_model_r2_train:.6f}\n--")
print(f"MSE(Ridge (L2)) on (Test) {ridge_model_mse_test:.6f}")
print(f"R2(Ridge (L2)) on (Test) {ridge_model_r2_test:.6f}\n")

# ==== Lasso ====
lasso_model = Lasso(alpha=0.001, random_state=42, max_iter=4_000)
lasso_model.fit(Xp_train, y_train)

lasso_model_mse_train = mean_squared_error(y_train, lasso_model.predict(Xp_train))
lasso_model_mse_test = mean_squared_error(y_test, lasso_model.predict(Xp_test))
lasso_model_r2_train = r2_score(y_train, lasso_model.predict(Xp_train))
lasso_model_r2_test = r2_score(y_test, lasso_model.predict(Xp_test))

print("==== Lasso (L1) ====")
print(f"MSE(Lasso (L1)) on (Train) {lasso_model_mse_train:.6f}")
print(f"R2(Lasso (L1)) on (Train) {lasso_model_r2_train:.6f}\n--")
print(f"MSE(Lasso (L1)) on (Test) {lasso_model_mse_test:.6f}")
print(f"R2(Lasso (L1)) on (Test) {lasso_model_r2_test:.6f}\n")
# === ================================ ===
