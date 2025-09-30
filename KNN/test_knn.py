from sklearn import datasets
from knn import KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(accuracy_score(y_test, predictions))


plt.figure()
plt.scatter(X_test[:,0], X_test[:,1], c=predictions, cmap='winter')
plt.show()
