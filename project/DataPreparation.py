import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    .drop('timestamp', axis=1)
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

data_g = full_data.copy()
# ---------------

# --- Deleting movies, that appear less than 20 times ---
movie_counts = full_data['item_id'].value_counts()
full_data = full_data[full_data['item_id'].isin(movie_counts[movie_counts >= 20].index)]
# -------------------------------------------------------

full_data = full_data.drop(['movie_id', 'item_id', 'user_id'], axis=1) # No need for them

occupation_dummies = pd.get_dummies(full_data['occupation'], dtype=int) # Making occupation encoding as it done with movie genres
full_data = pd.concat([full_data, occupation_dummies], axis=1).drop('occupation', axis=1)

# --- Normalization ---
u_norm = full_data.copy()
cols_to_normalize = ['age', 'release_year']

min_vals = u_norm[cols_to_normalize].min()
max_vals = u_norm[cols_to_normalize].max()
u_norm[cols_to_normalize] = (u_norm[cols_to_normalize] - min_vals) / (max_vals - min_vals)
# ---------------------

X = u_norm.drop('rating', axis=1).values
X = np.c_[np.ones(X.shape[0]), X]

y = u_norm['rating'].values


u_norm.to_csv("MovieSet.csv", index=False)
np.savez_compressed('MovieSet.npz', X=X, y=y)
print("--SUCCESS!--")



# ======= Data Visualisation =======
fig = plt.figure(figsize=(22, 22))
grid = (3, 3)

# == Number of Each Rating ===
labels_r = data_g['rating']
count_r = [len(labels_r[labels_r == 1]), len(labels_r[labels_r == 2]),
           len(labels_r[labels_r == 3]), len(labels_r[labels_r == 4]),
           len(labels_r[labels_r == 5])]

user_group = data_g.groupby('user_id')['age']

ax1 = plt.subplot2grid(grid, (0, 0))
ax1.bar([1, 2, 3, 4, 5], count_r, color='skyblue', edgecolor='black')
ax1.set_xlabel('Rating (Stars)')
ax1.set_ylabel('NUmber of rating')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
# ----------------------------------------------------


# == Users Activity and Movie Popularity ==
user_activity = data_g.groupby('user_id')['rating'].count()

ax2 = plt.subplot2grid(grid, (0, 1))
ax2.hist(user_activity, bins=50, color='skyblue', edgecolor='black')
ax2.set_title('Number of ratings per user')
ax2.set_xlabel('Number of ratings')
ax2.set_ylabel('Number of users')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

movie_popularity = data_g.groupby('item_id')['rating'].count()

ax3 = plt.subplot2grid(grid, (0, 2))
ax3.hist(movie_popularity, bins=50, color='skyblue', edgecolor='black')
ax3.set_title('Number of ratings per movie')
ax3.set_xlabel('Number of ratings')
ax3.set_ylabel('Number of movies')
ax3.grid(axis='y', linestyle='--', alpha=0.7)
# ----------------------------------------------------


# == Mean of user/movie ratings ==
user_mean_rating = data_g.groupby('user_id')['rating'].mean()

ax4 = plt.subplot2grid(grid, (1, 0))
ax4.hist(user_mean_rating, bins=30, color='coral', edgecolor='black')
ax4.set_title('Mean user rating distribution')
ax4.set_xlabel('Mean rating')
ax4.set_ylabel('Number of users')
ax4.grid(axis='y', linestyle='--', alpha=0.7)

movie_mean_rating = data_g.groupby('item_id')['rating'].mean()

ax5 = plt.subplot2grid(grid, (1, 1))
ax5.hist(movie_mean_rating, bins=30, color='gold', edgecolor='black')
ax5.set_title('Mean movie rating distribution')
ax5.set_xlabel('Mean rating')
ax5.set_ylabel('Number of movies')
ax5.grid(axis='y', linestyle='--', alpha=0.7)
# ----------------------------------------------------


# == Number of Ratings per Genre ==
genres = data_g.iloc[:, 7:-1].columns
genres_in_df = [g for g in genres if g in data_g.columns]
if genres_in_df:
    genre_counts = data_g[genres_in_df].sum().sort_values(ascending=False)

    ax6 = plt.subplot2grid(grid, (1, 2))
    ax6.barh(genre_counts.index, genre_counts.values, color='teal', edgecolor='black')
    ax6.set_title('Number of rating per genre')
    ax6.set_xlabel('Number of ratings')
    ax6.set_ylabel('Genre')
    ax6.tick_params(axis='x', rotation=75)
    ax6.grid(axis='x', linestyle='--', alpha=0.7)
# ----------------------------------------------------


# == User Age Distribution ==
unique_users = data_g.drop_duplicates(subset=['user_id'])

ax7 = plt.subplot2grid(grid, (2, 0))
ax7.hist(unique_users['age'], bins=20, color='purple', edgecolor='black', alpha=0.7)
ax7.set_title('User Age Distribution')
ax7.set_xlabel('Age')
ax7.set_ylabel('Number of users')
ax7.grid(axis='y', linestyle='--', alpha=0.7)
# ----------------------------------------------------


# == Age distribution depending on the rating given ==
ratings = sorted(data_g['rating'].unique())
data_to_plot = [data_g[data_g['rating'] == r]['age'] for r in ratings]

ax8 = plt.subplot2grid(grid, (2, 1), colspan=2)

box = ax8.boxplot(data_to_plot, patch_artist=True)
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
ax8.set_xticklabels(ratings)
ax8.set_title('Age distribution depending on the rating given')
ax8.set_xlabel('Given Rating')
ax8.set_ylabel('User Age')
ax8.grid(axis='y', linestyle='--', alpha=0.7)
# ----------------------------------------------------


plt.tight_layout(pad=2.0)
plt.savefig('EDA/DataRaw.svg')
plt.savefig('EDA/DataRaw.png')
plt.show()