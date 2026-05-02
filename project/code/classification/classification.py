import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = np.load('../../MovieSet.npz')
X = data['X']
y = data['y']

y_bin = np.where(y >= 4, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_bin, 
    test_size=0.2, 
    random_state=42,
    stratify=y_bin
)

clf = LogisticRegression(max_iter=4000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].bar(['Bad', 'Good'], [len(y_bin[y_bin == 0]), len(y_bin[y_bin == 1])], color=['#D14747', "#6ED147"])
ax[0].set_ylabel('Amount')
ax[0].set_xlabel('Rating Class')

ax[1].bar([1, 2, 3, 4, 5], [len(y[y == 1]), len(y[y == 2]), len(y[y == 3]), len(y[y == 4]), len(y[y == 5])], color='skyblue', edgecolor='black')
ax[1].set_xlabel('Rating (Stars)')

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred, 
    display_labels=['Bad', 'Good'], 
    cmap=plt.cm.Blues, 
    ax=ax[2]             
)
disp.ax_.set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig('../../EDA/classfication_results.png')
plt.show()


print(y[1116])
print(clf.predict(X[1116].reshape(1, -1)))