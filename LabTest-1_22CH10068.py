# NAME : SOHAM PAL
# ROLL NO : 22CH10068

# Importing libraries that will be used later
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Function to calculate accuracy :
def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_class = (y_pred > 0.5).astype(int)
    correct = np.sum(y_true == y_pred_class)
    return (correct / len(y_true)) * 100


# Implementation of closed form linear regression:
class linear_regression_closed:
    def __init__(self):
        self.weights = None

    def Train(self, X_train, y_train):
        X_w_intercept = np.column_stack((np.ones(X_train.shape[0]), X_train))
        #       Using the formula to calculate weights in closed form :
        self.weights = np.dot((np.linalg.inv(np.dot(X_w_intercept.T, X_w_intercept))), np.dot(X_w_intercept.T, y_train))
    
    def Predict(self, X_test):
        X_w_intercept = np.column_stack((np.ones(X_test.shape[0]), X_test))

        return np.dot(X_w_intercept, self.weights)


# Implementation of gradient descent linear regression:
class linear_regression_gradient:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None

    def Train(self, X_train, y_train):
        X_w_intercept = np.column_stack((np.ones(X_train.shape[0]), X_train))
        self.weights = np.zeros(X_w_intercept.shape[1])

        #       Gradient descent algorithm
        for _ in range(self.n_iter):
            pred = np.dot(X_w_intercept, self.weights)
            error = pred - y_train
            grad = np.dot(X_w_intercept.T, error) / len(y_train)
            self.weights -= self.lr * grad

    def Predict(self, X_test):
        X_w_intercept = np.column_stack((np.ones(X_test.shape[0]), X_test))

        return np.dot(X_w_intercept, self.weights)


# Experiment 1:
# Loading the Diabetes Dateset and deleting the unwanted columns
file_path = 'https://drive.google.com/file/d/1DnNNjyv4lbLHBMxC74C3kzPbhdFwxOWE/view?usp=drive_link'
file_path = 'https://drive.google.com/uc?id=' + file_path.split('/')[-2]
Diabetes_data = pd.read_csv(file_path)
# Making the altered data set
dataset_altered = Diabetes_data
dataset_altered = dataset_altered.drop(columns=['Pregnancies', 'SkinThickness', 'DiabetesPedigreeFunction'])
print("The first 10 rows of the alternated dataset are:")
print(dataset_altered.head(10))

# Experiment 2:
# Correlation matrix:
print(dataset_altered.corr())
sns.heatmap(dataset_altered.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix')
plt.show()
print("From the correlation matrix heatmap we can say that 'Glucose' and 'Outcome' are strongly correlated ")

# Experiment 3:
dataset_altered_features = dataset_altered.drop(columns=['Outcome'])
dataset_altered_targets = dataset_altered['Outcome']
# Splitting the data set into training and testing subsets( With 80 : 20 ratio):
X_train, X_test, y_train, y_test = train_test_split(dataset_altered_features, dataset_altered_targets, train_size=0.8,
                                                    shuffle=True, random_state=100)

model = linear_regression_closed()
model.Train(X_train, y_train)

y_pred_test = model.Predict(X_test)
y_pred_train = model.Predict(X_train)

print("The Percentage accuracy on training data :")
print(accuracy(y_train, y_pred_train))

print("The Percentage accuracy on testing data :")
print(accuracy(y_test, y_pred_test))

# Printing the confusion matrix:
y_pred_test_bi = [1 if x > 0.5 else 0 for x in y_pred_test]
cm = confusion_matrix(y_test , y_pred_test_bi)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Experiment 4 :

print("Accuracy for different learning rates :")
# A) learning rate = 0.00001:
model = linear_regression_gradient(0.00001, 50)
model.Train(X_train, y_train)
y_pred_test_a = model.Predict(X_test)

print("LR = 0.00001")
print(accuracy(y_test, y_pred_test_a))
# B) learning rate = 0.001:
model = linear_regression_gradient(0.001, 50)
model.Train(X_train, y_train)
y_pred_test_b = model.Predict(X_test)

print("LR = 0.001")
print(accuracy(y_test, y_pred_test_b))
# C) learning rate = 0.05:
model = linear_regression_gradient(0.05, 50)
model.Train(X_train, y_train)
y_pred_test_c = model.Predict(X_test)

print("LR = 0.05")
print(accuracy(y_test, y_pred_test_c))
# D) learning rate = 0.1:
model = linear_regression_gradient(0.1, 50)
model.Train(X_train, y_train)
y_pred_test_d = model.Predict(X_test)

print("LR = 0.1")
print(accuracy(y_test, y_pred_test_d))

print("Hence the optimal learning rate is 0.00001")

model = linear_regression_gradient(0.00001, 50)
model.Train(X_train, y_train)
y_pred_test_a = model.Predict(X_test)
y_pred_train_a = model.Predict(X_train)

print("The Percentage accuracy on training data :")
print(accuracy(y_train, y_pred_train_a))

print("The Percentage accuracy on testing data :")
print(accuracy(y_test, y_pred_test_a))

# Printing the confusion matrix:
y_pred_test_a_bi = [1 if x > 0.5 else 0 for x in y_pred_test_a]
cm = confusion_matrix(y_test , y_pred_test_a_bi)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
