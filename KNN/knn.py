import numpy as np
from collections import Counter


def euclidean_distance(x1 , x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

class KNN:
    def __init__(self , k=3):
        self.k = k
    

    def fit(self , X, y):
        self.X_train = X
        self.y_train = y

    def predict(self , X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)


    def _predict(self, x):
        # compute the distance
        distance = [euclidean_distance(x , x_train) for x_train in self.X_train]
        # get the closest samples and labels
        k_indices = np.argsort(distance)[:self.k]
        labels = [self.y_train[i] for i in k_indices]
        # majority vote
        most_common = Counter(labels).most_common(1)[0][0]

        return most_common
 