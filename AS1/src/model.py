import numpy as np

def regression_error(y_hat,y):
    return 1/y.shape[0] * np.sum((y_hat-y)**2)

class LinearRegressionIterative:
    def __init__(self):
        self.history = {}
        
    def fit(self, X, y, epochs, learning_rate=0.00001, verbose = 0):
        assert X.shape[0] == y.shape[0]
        num_samples, num_features = X.shape
        # Init weights to 0, account for bias also
        self.theta = np.zeros(num_features+1)
        
        # Add x_0 feature array (1s) to X
        x_0 = np.ones(num_samples).reshape((num_samples,-1))
        Z = np.hstack((x_0, X))
        
        # Start epoch and train
        for epoch in range(epochs):
            y_hat = self.predict(Z)#.flatten()
            J_theta = 1/2 * (y_hat - y).T.dot((y_hat - y))
            if verbose:
                print(f"Epoch {epoch}: loss = {J_theta}")
            
            grad_J_theta = Z.T.dot((y_hat - y))
            self.theta = self.theta - learning_rate*grad_J_theta
            
    
    def predict(self, X):
        if X.shape[1] == self.theta.shape[0]:
            # Bias is already in both theta and X
            return X.dot(self.theta)
        else:
            num_samples, num_features = X.shape
            x_0 = np.ones(num_samples).reshape((num_samples,-1))
            Z = np.hstack((x_0, X))
            return Z.dot(self.theta)
        
class LinearRegressionExplicit:
    def __init__(self):
        self.history = {}
        
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        num_samples, num_features = X.shape
        #y = y.reshape((num_samples,-1))
        # Add x_0 feature array (1s) to X
        x_0 = np.ones(num_samples).reshape((num_samples,-1))
        Z = np.hstack((x_0, X))
        #Z = np.insert(X, 0, 1, axis=1)
        self.theta = np.linalg.pinv(Z.T.dot(Z)).dot(Z.T).dot(y)
            
    
    def predict(self, X):
        num_samples, num_features = X.shape
        x_0 = np.ones(num_samples).reshape((num_samples,-1))
        Z = np.hstack((x_0, X))
        #Z = np.insert(X, 0, 1, axis=1)
        return Z.dot(self.theta)
    
class LinearRegressionDual:
    def __init__(self):
        self.history = {}
    
    def _magnitude(self,a):
        return np.sqrt(np.sum(a**2))
    
    def _gaussian_kernel(self,x1,x2,sigma=5):
        return np.exp(-(magnitude(x1-x2)**2)/(2*sigma**2))
    
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        num_samples, num_features = X.shape
        y = y.reshape((num_samples,-1))
        
        # Compute gram matrix using gaussian kernel
        G = pairwise_distances(x,metric=self._gaussian_kernel)
        assert G.shape[0] == G.shape[1] == X.shape[0]
        
        # Solve for alpha
        self.alpha = np.linalg.pinv(G.T.dot(G)).dot(G.T).dot(y)
    
    def predict(self, X):
        pass