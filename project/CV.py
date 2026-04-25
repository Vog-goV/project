import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    SGDRegressor
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

os.chdir(os.path.dirname(os.path.abspath(__file__)))


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

data_df = pd.read_csv("MovieSet_MODIFIED.csv")
target_col = 'rating'
feature_cols = [c for c in data_df.columns if c != target_col]
print(f"\nData Shape: {data_df.shape}")
print(f"Features: {feature_cols}")
# print(data_df.head(), '\n')

data = np.load('MovieSet_MODIFIED.npz')

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# ===========================
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples\n")


pipeline = Pipeline([
    ('sgd', SGDRegressor(loss='squared_error', max_iter=10000, learning_rate='adaptive', random_state=42))
])

param_grid = {
    'sgd__alpha': [0.0001, 0.001, 0.01, 0.1, 10], 
    'sgd__eta0': [0.0001, 0.001, 0.01, 0.1],       
    'sgd__penalty': ['l2', 'l1', 'elasticnet']              
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5, 
    scoring='neg_mean_squared_error', 
    n_jobs=-1,                         
    verbose=2,                 
)

grid_search.fit(X_train, y_train)

print(f"\nBest params: {grid_search.best_params_}\n")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nMSE on best: {mean_squared_error(y_test, y_pred):.6f}")
# === ================================ ===

print(y_test[1116])
print(best_model.predict(X_test[1116].reshape(1, -1)))