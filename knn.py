import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_label = [self._predict(x) for x in X]
        return np.array(predicted_label)

    def _predict(self, x):
        #compute distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #get k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_label = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_label).most_common(1)
        return most_common[0][0]

