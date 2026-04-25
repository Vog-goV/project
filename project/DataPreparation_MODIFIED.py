import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_COLS = ['user_id', 'item_id', 'rating', 'timestamp']
USER_COLS = ['user_id', 'age', 'gender', 'occupation', 'zip code']
ITEM_COLS = [
    'movie_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# ---- Loading Sets ----
u_data = (
    pd.read_csv('ml-100k/u.data', sep='\t', names=DATA_COLS)
)

u_item = (
    pd.read_csv('ml-100k/u.item', sep='|', names=ITEM_COLS, encoding='latin-1')
    .drop(['IMDb URL', 'video release date', 'movie title'], axis=1)
    .dropna()
)
release_years = pd.to_datetime(u_item.pop('release date')).dt.year # Only years in date should stay
u_item.insert(u_item.columns.get_loc('unknown'), 'release_year', release_years) 

u_user = (
    pd.read_csv('ml-100k/u.user', sep='|', names=USER_COLS)
    .drop('zip code', axis=1)
)
u_user['gender'] = u_user['gender'].map({'F': 0, 'M': 1}) # ---- Male -> 1 | Female -> 0
# ----------------------


# --- Merging ---
ratings_movies = pd.merge(u_data, u_item, left_on='item_id', right_on='movie_id', how='inner')
full_data = pd.merge(ratings_movies, u_user, on='user_id', how='inner')

cols = full_data.columns
full_data = full_data[
    list(cols[:1]) + 
    list(cols[-3:-1]) + 
    [c for c in cols if c not in cols[:1] and c not in cols[-3:-1]]
]
# ---------------


# --- Deleting movies, that appear less than 20 times ---
movie_counts = full_data['item_id'].value_counts()
full_data = full_data[full_data['item_id'].isin(movie_counts[movie_counts >= 20].index)]
# -------------------------------------------------------

# One Hot Encoding for occupations
occupation_dummies = pd.get_dummies(full_data['occupation'], dtype=int)
full_data = pd.concat([full_data, occupation_dummies], axis=1).drop('occupation', axis=1)
# --------------------------------

# ---- Adding 2 new Features and Splitting Dataset
train_data, test_data = train_test_split(full_data, random_state=42, test_size=0.2) 

user_mean_train = train_data.groupby("user_id")['rating'].mean().rename('user_mean_rating')
movie_mean_train = train_data.groupby("item_id")['rating'].mean().rename('movie_mean_rating')

train_data = train_data.merge(user_mean_train, on='user_id', how='left').merge(movie_mean_train, on='item_id', how='left')
test_data = test_data.merge(user_mean_train, on='user_id', how='left').merge(movie_mean_train, on='item_id', how='left')


# === 2 different set of features to drop
cols_to_drop = ['item_id', 'movie_id', 'user_id', 'timestamp'] # Leaves 46 features
# cols_to_drop = (full_data.iloc[:, 8:-1].columns).to_list() + ['item_id', 'movie_id', 'user_id', 'timestamp'] # dropping occupation, genres and useless data # Leaves 7 features
# ---------------------------------------

train_data = train_data.drop(cols_to_drop, axis=1)
test_data = test_data.drop(cols_to_drop, axis=1)

cols_to_normalize = ['age', 'release_year', 'user_mean_rating', 'movie_mean_rating']
min_vals = train_data[cols_to_normalize].min()
max_vals = train_data[cols_to_normalize].max()
train_data[cols_to_normalize] = (train_data[cols_to_normalize] - min_vals) / (max_vals - min_vals)
test_data[cols_to_normalize] = (test_data[cols_to_normalize] - min_vals) / (max_vals - min_vals)

y_train = train_data['rating'].values
X_train = train_data.drop('rating', axis=1).values
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

y_test = test_data['rating'].values
X_test = test_data.drop('rating', axis=1).values
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

pd.concat([train_data, test_data]).to_csv("MovieSet_MODIFIED.csv", index=False)

np.savez_compressed('MovieSet_MODIFIED.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("--SUCCESS!--")