import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None, plusplus=False):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.labels_ = None
        self.plusplus = plusplus
    
    def initialize_centroids_plusplus(self, X): 
        
        first_idx = np.random.choice(X.shape[0], replace = False)
        self.centroids = [X[first_idx]]
        
        # Assigning probabilities to each point based on distance from the nearest centroid
        for i in range(1, self.k):
            dists = []
            for x in X:
                min_dist = min([np.linalg.norm(x - c) for c in self.centroids])
                dists.append(min_dist)
                
            dists = np.array(dists)
            prob = dists**2 / np.sum(dists**2)
            new_c = np.random.choice(X.shape[0],replace = False, p=prob)
            self.centroids.append(X[new_c])

        self.centroids = np.array(self.centroids)
        self.first_cent = self.centroids.copy()
        
    
    def fit(self, X):
        """
        Fit the KMeans model to data X.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        
        self.data = X
        self.convergence = []
        
        self.initialize_centroids_plusplus(self.data)
        
        self.labels = np.zeros(self.data.shape[0], dtype=int)
        
        for i in range(self.max_iters):
            for i, point in enumerate(self.data):
                distances = np.sqrt(np.sum((point - self.centroids)**2, axis=1))
                self.labels[i] = np.argmin(distances)
                
            old_centroid = self.centroids.copy()
            
            for i in range(self.centroids.shape[0]):
                idx = np.where(self.labels == i)
                self.centroids[i] = np.mean(self.data[idx], axis=0)
            
            convergence_value = np.sqrt(np.sum((old_centroid - self.centroids)**2, axis=1))
            self.convergence.append(convergence_value)
            if np.max(convergence_value) <= self.tol:
                break
            
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        """
        # TODO: Implement this function
        self.predicted = X
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        
        for i, point in enumerate(X):
            distances = np.sum(np.sqrt((point - self.centroids)**2), axis=1)
            self.labels_[i] = np.argmin(distances)

    
    def fit_predict(self, X):
        """
        Perform KMeans clustering and return cluster labels.
        """
        self.fit(X)
        return self.labels_
    