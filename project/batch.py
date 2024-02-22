from sklearn.pipeline import Pipeline as sk_Pipeline
from sklearn.preprocessing import StandardScaler as sk_StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Callable 
import numpy as np 
import matplotlib.pyplot as plt



class SequentialImpute(BaseEstimator, TransformerMixin):
    def __init__(self, detect: Callable = lambda x: x == -200):
        self.detect = detect 
        self.last = None 
        self.last_target = None
        
    def fit(self, X, y):
        X, y = map(np.array, (X, y))
        self.last = np.zeros(X.shape[1])
        self.last_target = 0
        self.X_ = X 
        self.y_ = y 
        for i in range(len(y)):
            x = np.where(self.detect(X[i]), self.last, X[i])
            self.last = x 
            if not self.detect(y[i]):
                self.last_target = y[i]
        return self 
    
    def transform(self, X):
        X = np.array(X)
        for i, x in enumerate(X):
            X[i] = np.where(self.detect(x), self.last, x)
            self.last = x
        return X

def create_pipeline(model, impute: bool = False) -> sk_Pipeline:
    steps = []
    if impute:
        steps.append(SequentialImpute())
    steps = [('selector', SelectKBest(k = 10, score_func=f_regression)),
            ('scaler', sk_StandardScaler()), ('model', model)]    
    pipeline = sk_Pipeline(steps)
    return pipeline


def plot_model_performance(models, mae_values):
    plt.figure(figsize=(10, 6))
    plt.bar(models, mae_values, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('batch_learning_model_performance.png')
    plt.show()