import random
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import *
import numpy as np
import math

# HELPERS
# Weight init
def xavier_init(fan_in, fan_out, constant=1):
    """
    fan_in: number of rows
    fan_out: number of columns
    return: a random matrix of shape (fan_in,fan_out) with specific characteristics
    purpose: initialize weights and biases
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low, high,(fan_in, fan_out))

# Retain a subset of classes
def retain_subset(df, subsets=[0,4,6]):
    df['label'] = df['label'].apply(lambda x: x if x in subsets else -1)
    df = df[df.label!=-1]
    
    # Remap classes to string then back to integer for them to be [0,1,2] instead of [0,4,6] since we will be one hot encoding label => [0,4,6] will lead to inconsistent one hot encoding with final results 
    def class_map(label):
        if label == 0:
            return "tshirt"
        elif label == 4:
            return "coat"
        elif label == 6:
            return "shirt"
        return "none"       
    df['label'] = df['label'].apply(lambda x: class_map(x))
    assert df[df.label == "none"].shape[0] == 0
    return df

def get_kfold_results(X_train_normalized,y_train_ohe,y_train,hidden_units=None, learning_rate=None):
    kf = KFold(n_splits=5)
    train_accuracy = []
    test_accuracy = []
    num_class = len(np.unique(y_train))
    for train_index, val_index in kf.split(X_train_normalized):
        # Split to train and test for this fold
        X_train_kfold, X_val = X_train_normalized[train_index], X_train_normalized[val_index]
        y_train_kfold_ohe, y_val = y_train_ohe[train_index].toarray(), y_train[val_index]
        y_train_kfold = y_train[train_index]
        
        X_train_poly = X_train_kfold
        X_test_poly = X_val

        # Train the neural net
        if hidden_units:
            nn = TwoLayerNet(num_class,hidden_units=hidden_units)
        else:
            nn = TwoLayerNet(num_class,hidden_units=128)
        
        if learning_rate:
            nn.fit(X_train_poly,y_train_kfold_ohe,epochs=100,learning_rate=learning_rate)
        else:
            nn.fit(X_train_poly,y_train_kfold_ohe,epochs=100,learning_rate=1e-6)
            
        # Predict and error
        train_pred = nn.predict(X_train_poly)
        test_pred = nn.predict(X_test_poly)

        # Save the accuracy of this fold
        train_accuracy.append(accuracy_score(train_pred,y_train_kfold))
        test_accuracy.append(accuracy_score(test_pred,y_val))
        
    return train_accuracy, test_accuracy

# 2-CLASS LOGISTIC REGRESSION
class LogisticRegression2Class:
    def __init__(self):
        self.history = {}
        
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def fit(self, X, y, epochs, learning_rate=0.00001, verbose = 0):
        assert X.shape[0] == y.shape[0]
        num_samples, num_features = X.shape
        # Init weights to 0, account for bias also
        self.theta = np.random.randn(num_features+1)
        
        # Add x_0 feature array (1s) to X
        x_0 = np.ones(num_samples).reshape((num_samples,-1))
        Z = np.hstack((x_0, X))
        
        # Start epoch and train
        for epoch in range(epochs):
            h = self.predict_proba(Z)
            J_theta = -(np.sum((y*np.log(h)) + ((1-y)*np.log(1-h))))
            
            if verbose:
                print(f"Epoch {epoch}: loss = {J_theta}")
            
            grad_J_theta = Z.T.dot((h - y))
            self.theta = self.theta - learning_rate*grad_J_theta
    
    def predict(self,X):
        return np.round(self.predict_proba(X))
    
    def predict_proba(self,X):
        # If bias is already in both theta and X
        if X.shape[1] == self.theta.shape[0]:
            return self._sigmoid(X.dot(self.theta))
        else:
            num_samples, num_features = X.shape
            x_0 = np.ones(num_samples).reshape((num_samples,-1))
            Z = np.hstack((x_0, X))
            return self._sigmoid(Z.dot(self.theta))

# MULTI CLASS LOGISTIC REGRESSION
class LogisticRegressionMultiClass:
    def __init__(self,num_class):
        self.num_class = num_class
        self.history = {}
        
    def _softmax(self,x):
        """
        x: mxk input matrix (m is the number of instances, k is the number of classes)
        return: a mxk matrix containing softmax value for each k class for each example
        """
        res = np.exp(x).T/np.sum(np.exp(x),axis=1)
        return res.T
    
    def _categorical_cross_entropy(self,y,y_hat):
        """
        y: one-hot encoded target vector of shape mx3 (m is number of instances)
        y_hat: prediction/softmax output of shape mx3
        """
        # y being one-hot encoded will have similar effect to the indicator function in the original formula 
        # as we are multiplying by 0 if the current example is not of that class, and by 1 otherwise
        # Sum from numpy will sum across all axis
        return np.sum(y*np.log(y_hat))
    
    def fit(self, X, y, epochs, learning_rate=0.00001, verbose = 0):
        assert X.shape[0] == y.shape[0]
        try:
            assert y.shape[1] >= 2
        except AssertionError:
            print("y needs to be one-hot encoded")
        
        self.num_samples, num_features = X.shape
        
        # Init weights randomly, also account for bias in self.theta 
        self.theta = xavier_init(num_features+1,self.num_class)
        
        # Add x_0 feature array (1s) to X
        x_0 = np.ones(self.num_samples).reshape((self.num_samples,-1))
        Z = np.hstack((x_0, X))
        
        # Start epoch and train
        for epoch in range(epochs):
            y_hat = self.predict_proba(Z)
            J_theta = self._categorical_cross_entropy(y,y_hat)
            
            if verbose:
                print(f"Epoch {epoch}: loss = {J_theta}")
            
            grad_J_theta = Z.T.dot((y_hat - y))
            self.theta = self.theta - learning_rate*grad_J_theta
    
    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)
    
    def predict_proba(self,X):
        if X.shape[1] == self.theta.shape[0]:
            # Bias is already in both theta and X
            return self._softmax(X.dot(self.theta))
        else:
            num_samples, num_features = X.shape
            x_0 = np.ones(num_samples).reshape((num_samples,-1))
            Z = np.hstack((x_0, X))
            return self._softmax(Z.dot(self.theta))

# TWO LAYER NEURAL NET
class TwoLayerNet:
    def __init__(self,num_class,hidden_units = 64):
        self.history = {}
        self.num_class = num_class
        self.hidden_units = hidden_units
    
    def _softmax(self,x):
        """
        x: mx3 input matrix (m is the number of instances, 3 is the number of classes)
        return: a mx3 matrix containing softmax value for each k=3 class for each example
        """
        # Check for correct dimension
        res = np.exp(x).T/np.sum(np.exp(x),axis=1)
        return res.T
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def _categorical_cross_entropy(self,y,y_hat):
        """
        y: one-hot encoded target vector of shape mx3 (m is number of instances)
        y_hat: prediction/softmax output of shape mx3
        """
        return -np.sum(y*np.log(y_hat))
        
    def fit(self, X, y, epochs, learning_rate=0.00001, verbose = 0):
        try:
            assert y.shape[1] >= 2
        except AssertionError:
            print("y needs to be one-hot encoded")
        assert self.num_class == 3
        
        self.num_samples = X.shape[0]
        self.input_dim = X.shape[1]
        
        # Init weight via Xavier Init
        self.W1 = xavier_init(self.input_dim,self.hidden_units)
        self.W2 = xavier_init(self.hidden_units, self.num_class)
        
        for epoch in range(epochs):
            # Forward pass
            h = self._sigmoid(X.dot(self.W1))
            y_pred = self._softmax(h.dot(self.W2))
            J_theta = self._categorical_cross_entropy(y,y_pred)
            
            if verbose:
                print(f"Epoch {epoch}: loss = {J_theta}")
            
            # Backward pass
            grad_w2 = h.T.dot(y_pred-y)
            grad_w1 = X.T.dot(np.dot((y_pred-y),self.W2.T) * h * (1-h))
            
            self.W2 = self.W2 - learning_rate * grad_w2
            self.W1 = self.W1 - learning_rate * grad_w1
    
    def predict_proba(self,X):
        h = self._sigmoid(X.dot(self.W1))
        return self._softmax(h.dot(self.W2))
    
    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)