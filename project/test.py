import joblib
import numpy as np

data = np.load('models/7features_7degree.npz')

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

loaded_model = joblib.load('models/Poly_Ridge_7_features.pkl')
result = loaded_model.score(X_test, y_test)

print(y_test[1116])
print(loaded_model.predict(X_test[1116].reshape(1, -1)))