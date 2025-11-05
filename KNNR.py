import numpy as np

class KNNRegressor:
    """Your KNN Regressor implementation"""
    def __init__(self, k=3, p=1):
        self.k = k
        self.p = p 
        self.X_train = np.array([])
        self.y_train = np.array([])
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def Minkowski_d(self, y, X):
        return np.sum(np.abs(X - y) ** self.p, axis=1) ** (1 / self.p)
        
    def predict(self, X_test):
        predicted = []
        for test in X_test:
            distances = self.Minkowski_d(test, self.X_train) #collapse columns
            neighbors_indeces = np.argpartition(distances, self.k-1)
            values = np.sum(self.y_train[neighbors_indeces[:self.k]])
            predicted.append(values/self.k)   
        return predicted