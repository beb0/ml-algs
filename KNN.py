import numpy as np

class KNNClassifier:  
    """K-Nearest Neighbors classifier using only NumPy."""
    
    def __init__(self, k):
        """Initialize KNN classifier, providing k and the training data."""
        self.k = k
        self.X_train = np.array([])
        self.y_train = np.array([])
    
    def fit(self, X_train, y_train):
        """Fit the model using the training data."""
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        """Predict classes for multiple samples."""
        predicted = []
        for sample in X:
            distances = np.linalg.norm(self.X_train - sample, axis=1) #collapse columns
            neighbors_indeces = np.argpartition(distances, self.k-1)
            values, counts = np.unique(self.y_train[neighbors_indeces[:self.k]], return_counts=True)
            most_frequent = values[np.argmax(counts)]
            predicted.append(most_frequent)            
        return predicted