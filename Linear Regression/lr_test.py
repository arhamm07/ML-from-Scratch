from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
mse = mse(y_test, predictions)    
print("MSE: ", mse)

# Plot the model
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X_test, predictions, color='red', label='Predictions')
plt.show()