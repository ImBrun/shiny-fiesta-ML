import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

train = np.load("Datasets/fashion_train.npy")
test = np.load("Datasets/fashion_train.npy")

X_train = train[:, :784]
y_train = train[:, 784]

X_test = test[:, :784]
y_test = test[:, 784]


# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.fit_transform(X_test, y_test)

# Apply PCA for comparison
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test, y_test)

# Create and train a classification model using LDA-transformed features
model_lda = LogisticRegression(max_iter=20000)
model_lda.fit(X_lda, y_train)

# Create and train a classification model using PCA-transformed features
model_pca = LogisticRegression(max_iter=20000)
model_pca.fit(X_pca, y_train)

# Evaluate both models on the test data
y_pred_lda = model_lda.predict(X_test_lda)
y_pred_pca = model_pca.predict(X_test_pca)

# Calculate accuracy for both models
accuracy_lda = accuracy_score(y_test, y_pred_lda)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# Create confusion matrices
cm_lda = confusion_matrix(y_test, y_pred_lda)
cm_pca = confusion_matrix(y_test, y_pred_pca)

# Visualize confusion matrices
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix with LDA')

plt.subplot(1, 2, 2)
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix with PCA')

plt.tight_layout()
plt.show()