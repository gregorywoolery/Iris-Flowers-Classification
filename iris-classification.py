import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Steps Involved:
# Obtain a proper dataset for the task.
# Analyze and visualize the dataset, and try to gain an intuition about the model you should use.
# Convert the data into an usable form, and then preprocess it.
# Build the model, and evaluate it.
# Check where the model is falling short, and keep improving it.


# Data Preparation
# We import the required modules, load the dataset from sklearn’s iris
# datasets class in the form of a dictionary, get the features and targets
# into numpy arrays X and Y, respectively and the names of the classes into names.
dataset = load_iris()
X, Y, names = dataset['data'], dataset['target'], dataset['target_names']
target = dict(zip(np.array([0, 1, 2]), names))
iris = sns.load_dataset('iris')

sns.pairplot(iris, hue='species')
plt.show()

sns.heatmap(iris.corr(), annot=True, cmap='coolwarm')
plt.show()

# Preprocessing
# Introduce polynomial features into the model, like x₁², x₁x₂, x₂²
# to produce a non-linear decision boundary to better separate the classes.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
poly = PolynomialFeatures(degree=5)
poly.fit(X_train)
X_train_pr = poly.transform(X_train)
scaler = StandardScaler()
scaler.fit(X_train_pr)
X_train_pr = scaler.transform(X_train_pr)

# Create an object of the LogisticRegression class named model,
# and fit our training dataset to it
model = LogisticRegression()
model.fit(X_train_pr, Y_train)


# Use the predict function to generate the 1D array of predictions
# both on the training and test set
X_test_pr = poly.transform(X_test)
X_test_pr = scaler.transform(X_test_pr)
Y_train_pred, Y_test_pred = model.predict(X_train_pr), model.predict(X_test_pr)

print(accuracy_score(Y_train, Y_train_pred))
print(accuracy_score(Y_test, Y_test_pred))
print(classification_report(Y_test, Y_test_pred))
print(confusion_matrix(Y_test, Y_test_pred))

indices = np.random.randint(150, size=20)

# Note that Y_true represents the true class of the example and Y_pred
# is the predicted class.
X_pred, Y_true = X[indices], Y[indices]
X_pred_pr = poly.transform(X_pred)
X_pred_pr = scaler.transform(X_pred_pr)
Y_pred = model.predict(X_pred_pr)
target_true, target_pred = [], []

for i in range(len(Y_true)):
    target_true.append(target[Y_true[i]])
    target_pred.append(target[Y_pred[i]])

print("X Prediction")
print(X_pred)

print("Target True")
print(target_true)

print("Target Predictions")
print(target_pred)
