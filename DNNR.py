import numpy as np

class DNNRegressor:
    """DNNR - uses local gradients for Taylor approximation"""
    def __init__(self, k=5, p=2, k_gradient=10): 
        self.k = k  # neighbors for prediction
        self.p = p
        self.k_gradient = k_gradient  # neighbors for gradient estimation
        
    def fit(self, X_train, y_train):
        # TODO: Implement this function
        self.X_train = X_train
        self.y_train = y_train
    
    
    def estimate_gradient(self, X_m, y_m):
        distances = self.Minkowski_d(X_m, self.X_train) #collapse columns
        neighbors_indeces = np.argsort(distances)[:self.k_gradient+1]
        
        
        # Calculate gradient using least squares approximation for local neighbors
        X_neighbors = self.X_train[neighbors_indeces]
        y_neighbors = self.y_train[neighbors_indeces]
        
        a = X_neighbors - X_m
        b = y_neighbors - y_m
        
        gamma, *_ = np.linalg.lstsq(a, b, rcond=None)[0]
        
        return gamma
    
    def Minkowski_d(self, y, X):
        return np.sum(np.abs(X - y) ** self.p, axis=1) ** (1 / self.p)
    
    
    def predict(self, X_test):
        # TODO: Implement this function
        predicted = []
        for to_predict in X_test:
            
            distances = self.Minkowski_d(to_predict, self.X_train) #collapse columns
            neighbors_indeces = np.argpartition(distances, self.k-1)[:self.k]
            
            taylor_preds = []  
                  
            for x in neighbors_indeces:
                gradient = self.estimate_gradient(self.X_train[x], self.y_train[x])
                
                # Taylor expansion to account for local changes
                taylor_exp = self.y_train[x] + np.dot(gradient, (to_predict - self.X_train[x]))
                taylor_preds.append(taylor_exp)
                
            avg_pred = np.mean(taylor_preds)
            
            predicted.append(avg_pred)
        return predicted