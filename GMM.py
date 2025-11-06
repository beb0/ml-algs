import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    """
    Gaussian Mixture Model implementation using Expectation-Maximization algorithm.
    """
    def __init__(self, K, max_iters=100, tol=1e-4, regularize=False):
        self.means = None
        self.covariances = None
        self.weights = None
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.regularize = regularize
    
    def initialize_parameters(self, X):
        """Initialize GMM parameters."""
        n_samples, n_features = X.shape
        self.means = X[np.random.choice(n_samples, self.K, False)] 
        self.covariances = np.array([np.eye(n_features)] * self.K)
        # Initialize weights uniformly (coefficients sum to 1)
        self.weights = np.ones(self.K) / self.K
        
    def regularize_covariances(self, covariances, epsilon=1e-6):
        
        for k in range(self.K):
            covariances[k] += epsilon * np.eye(covariances[k].shape[0])
        return covariances
        
    def _e_step(self, X):
        N, D = X.shape
        K = self.K
        
        responsibilities = np.empty((N, K))
        
        #PDF 
        for k in range(K):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k])
        responsibilities_sum = responsibilities.sum(axis=1)[:, np.newaxis]
        
        responsibilities /= responsibilities_sum
        # print(responsibilities.sum(axis=1))
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """Maximization step"""
        N, D = X.shape
        N_k = np.sum(responsibilities, axis=0)
        
        
        # Just Maximum Likelihood Estimation 
        for k in range(self.K):
            self.means[k] = np.sum(responsibilities[:, k][:, np.newaxis] * X, axis=0) / N_k[k]
            diff = X - self.means[k]
            self.covariances[k] = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / N_k[k]
            self.weights[k] = N_k[k] / N
        
        if self.regularize:
            self.covariances = self.regularize_covariances(self.covariances)
    
    def fit(self, X):
        """Fit GMM to data using EM algorithm."""
        # TODO: Implement this function
        self.initialize_parameters(X)  
        
        for _ in range(100): 
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            
        
        print("Means:", self.means)
        print("Covariances:", self.covariances)
        print("Weights:", self.weights)
        
    
    def predict(self, X):
        """
        Predict the component label for each sample.
        """
        
        # Just calculating the pdf for each component and assigning 
        # the max as we have already calculated the parameters in fit method
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)