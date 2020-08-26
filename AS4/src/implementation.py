import numpy as np
import cvxopt
from numpy import linalg
import matplotlib.pyplot as plt

def plot_support_vectors(X, y, support_vectors, SVM_model):
    plt.scatter(X[:,0],X[:,1], c=y, s=30, cmap = plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))

    Z = SVM_model.decision_function(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                    linestyles=['--', '-', '--'])

    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
    #plt.show()
    
class HardMarginSVM:
    def __init__(self):
        self.W = 0
        self.W0 = 0
        self.alphas = []
        
    def fit(self,X,y,epsilon = 1e-6):
        m, n = X.shape
        # Reshape to perform row-wise multiplication
        y_reshaped = y.reshape(-1,1)
        
        # Convert to right format for quadratic programming solver
        # P = yy.T XX.T = y*X dot y*X.T
        P = cvxopt.matrix(np.dot(y_reshaped*X, (y_reshaped*X).T))
        # q = [-1...-1]
        q = cvxopt.matrix(np.ones((m,1))*-1)
        # A = y.T
        A = cvxopt.matrix(y_reshaped.T)
        # b = 0
        b = cvxopt.matrix(np.array([0.]))
        # G = diagonal of size mxm with -1s on since we are using hard margin, not soft margin
        G = cvxopt.matrix(np.eye(m)*-1)
        # h = [0...0] since our only constraint is alpha >= 0
        h = cvxopt.matrix(np.zeros(m))
        
        # Run solver to compute alphas
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        self.alphas = np.array(sol['x']).flatten()
        assert self.alphas.shape[0] == m
        
        # Compute W from alphas
        # W has shape (n,1) after computation
        self.W = (y*self.alphas).reshape(1,-1).dot(X).T#.flatten()
        
        # Compute W0
        # Get all support vectors: instances with alpha > epsilon
        sv = np.where(self.alphas > epsilon)[0]
        y_reshaped_sv = y_reshaped[sv]
        X_sv = X[sv]
        # Compute
        self.W0 = (1/len(sv)) * np.sum(y_reshaped_sv - X_sv.dot(self.W))
        
    def predict(self,X):
        res = np.sign(X.dot(self.W) + self.W0)
        # Convert 0 to -1 and return the result only containing class -1 or 1
        return np.apply_along_axis(self._convert_0,1,res)
    
    def decision_function(self,X):
        return X.dot(self.W) + self.W0
    
    def _convert_0(self,x):
        if x == 0:
            return -1
        return x
    
class SoftMarginSVM:
    def __init__(self):
        self.W = 0
        self.W0 = 0
        self.alphas = []
        
    def fit(self,X,y,C=10,epsilon = 1e-6):
        m, n = X.shape
        # Reshape to perform row-wise multiplication
        y_reshaped = y.reshape(-1,1)
        
        # Convert to right format for quadratic programming solver
        # P = yy.T XX.T = y*X dot y*X.T
        P = cvxopt.matrix(np.dot(y_reshaped*X, (y_reshaped*X).T))
        # q = [-1...-1]
        q = cvxopt.matrix(np.ones((m,1))*-1)
        # A = y.T
        A = cvxopt.matrix(y_reshaped.T)
        # b = 0
        b = cvxopt.matrix(np.array([0.]))
        # G = diagonal of size 2mxm as we are using soft margin
        G_1 = np.eye(m)*-1
        G_2 = np.eye(m)
        G = np.vstack((G_1, G_2))
        assert G.shape[0] == 2*m
        G = cvxopt.matrix(G)
        # h = [0...0, C...C] as we are using soft margin
        h_1 = np.zeros(m)
        h_2 = np.array([C]*m)
        h = np.hstack((h_1,h_2))
        assert h.shape[0] == 2*m
        h = cvxopt.matrix(h)
        
        # Run solver to compute alphas
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        self.alphas = np.array(sol['x']).flatten()
        assert self.alphas.shape[0] == m
        
        # Compute W from alphas
        # W has shape (n,1) after computation
        self.W = (y*self.alphas).reshape(1,-1).dot(X).T#.flatten()
        
        # Compute W0
        # Get all support vectors: instances with alpha > epsilon
        sv = np.where(self.alphas > epsilon)[0]
        y_reshaped_sv = y_reshaped[sv]
        X_sv = X[sv]
        # Compute
        self.W0 = (1/len(sv)) * np.sum(y_reshaped_sv - X_sv.dot(self.W))
        
    def predict(self,X):
        res = np.sign(X.dot(self.W) + self.W0)
        # Convert 0 to -1 and return the result only containing class -1 or 1
        return np.apply_along_axis(self._convert_0,1,res)
    
    def decision_function(self,X):
        return X.dot(self.W) + self.W0
    
    def _convert_0(self,x):
        if x == 0:
            return -1
        return x
    
class KernelSVM:
    def __init__(self, kernel = "polynomial", C = 10, epsilon = 1e-6, degree = 2, sigma = 5.0):
        self.W = 0
        self.W0 = 0
        self.alphas = []
        kernel_map = {
            "polynomial": self._polynomial_kernel,
            "radial": self._gaussian_kernel
        }
        self.kernel = kernel_map[kernel]
        self.C = C
        self.epsilon = epsilon
        self.degree = degree
        self.sigma = sigma
    
    def _polynomial_kernel(self,x1,x2):
        try:
            assert self.degree != 0
            return (np.dot(x1,x2)+1)**self.degree
        except:
            print("Please input degree > 0! Terminating...")
    
    def _gaussian_kernel(self,x1,x2):
        return np.exp((-linalg.norm(x1-x2)**2)/(2*(self.sigma**2)))
    
    def _build_gram(self,X):
        m, n = X.shape
        gram_matrix = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                gram_matrix[i,j] = self.kernel(X[i],X[j])
        return gram_matrix        
        
    def fit(self, X, y):
        m, n = X.shape
        # Reshape to perform row-wise multiplication
        y_reshaped = y.reshape(-1,1)
        
        # Explicitly build the gram matrix since we change the kernel here
        Gram = self._build_gram(X)
        # Convert to right format for quadratic programming solver
        # P = yy.T XX.T
        # Use np.outer to calculate the outer product, so we don't have to reshape and calculate like earlier implementation, since this time we multiply explicitly with a Gram matrix
        P = cvxopt.matrix(np.outer(y,y)*Gram)
        # q = [-1...-1]
        q = cvxopt.matrix(np.ones((m,1))*-1)
        # A = y.T
        A = cvxopt.matrix(y_reshaped.T.astype(float))
        # b = 0
        b = cvxopt.matrix(np.array([0.]))
        # G = diagonal of size 2mxm as we are using soft margin
        G_1 = np.eye(m)*-1
        G_2 = np.eye(m)
        G = np.vstack((G_1, G_2))
        assert G.shape[0] == 2*m
        G = cvxopt.matrix(G)
        # h = [0...0, C...C] as we are using soft margin
        h_1 = np.zeros(m)
        h_2 = np.array([self.C]*m)
        h = np.hstack((h_1,h_2))
        assert h.shape[0] == 2*m
        h = cvxopt.matrix(h)
        
        # Run solver to compute alphas
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        self.alphas = np.array(sol['x']).flatten()
        assert self.alphas.shape[0] == m
        
        # Compute W0
        # Get all support vectors: instances with alpha > epsilon
        sv = np.where(self.alphas > self.epsilon)[0]
        y_reshaped_sv = y_reshaped[sv]
        self.y_sv = y[sv]
        self.X_sv = X[sv]
        self.alphas_sv = self.alphas[sv]
        # Compute W0 using the new formula involving gram matrix since we no longer in the dimensional space of X
        self.W0 = 0
        ind = np.arange(len(self.alphas))[sv]
        for i in range(len(self.alphas_sv)):
            self.W0 += self.y_sv[i]
            self.W0 -= np.sum(self.alphas_sv * self.y_sv * Gram[ind[i],sv])
        self.W0 /= len(self.alphas_sv)
        
    def predict(self,X):
        kernel_results = np.sign(self.decision_function(X))
        
        # Convert 0 to -1 and return the result only containing class -1 or 1
        return np.apply_along_axis(self._convert_0,1,kernel_results)
    
    def decision_function(self,X):
        kernel_results = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            res = []
            for j in range(self.X_sv.shape[0]):
                res.append(self.y_sv[j]*self.alphas_sv[j]*self.kernel(self.X_sv[j],X[i]))
            
            kernel_results[i] = np.sum(res) + self.W0
        return kernel_results.reshape((-1,1))
    
    def _convert_0(self,x):
        if x == 0:
            return -1
        return x