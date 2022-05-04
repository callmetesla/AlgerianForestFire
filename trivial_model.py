import numpy as np
import pandas as pd
from sklearn import metrics
np.random.seed(69)

class TrivialModel(object):
    def __init__(self):
        self.probability = None
        self.n = None

    def fit(self, train_X, train_Y):
        train_df = pd.concat([train_X, train_Y], axis=1)
        num_class0 = train_df[train_df['Classes'] == 0].shape[0]
        num_class1 = train_df[train_df['Classes'] == 1].shape[0]
        self.n = num_class0 + num_class1
        self.probability = num_class0 / self.n

    def predict(self, X):
        predictions = np.random.binomial(1, self.probability, size=X.shape[0])
        return predictions
    
    def compute_scores(self, predictions, Y):
        ground_truths = list(Y)
        accuracy = metrics.accuracy_score(ground_truths, predictions)
        F1 = metrics.f1_score(ground_truths, predictions)
        confusion_matrix = metrics.confusion_matrix(ground_truths, predictions)
        results = {'F1': F1, 'accuracy': accuracy, 'confusion_matrix': confusion_matrix}
        print(results)
        return results
