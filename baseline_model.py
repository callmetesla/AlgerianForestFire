import pandas as pd
import numpy as np
from sklearn import metrics
from data_loader import normalize_columns_fn

class NearestMeans(object):
    def __init__(self):
        self.means = []
    
    def compute_means(self, train_X, train_Y):
        train_X = normalize_columns_fn(train_X)
        train_df = pd.concat([train_X, train_Y], axis=1)
        class_0_rows = train_df[train_df['Classes'] == 0].drop(columns=['Classes'])
        class_1_rows = train_df[train_df['Classes'] == 1].drop(columns=['Classes'])
        class_0_means = list(class_0_rows.mean())
        self.means.append(class_0_means)
        class_1_means = list(class_1_rows.mean())
        self.means.append(class_1_means)
        self.means = np.array(self.means)

    def fit(self, train_X, train_Y):
        self.compute_means(train_X, train_Y)

    def compute_distance(self, x, y):
        return np.linalg.norm(x - y)

    def get_min_distance_class(self, x, num_classes=2):
        distances = []
        for i in range(num_classes):
            distances.append(np.linalg.norm(x-self.means[i]))
        return np.argmin(distances)

    def predict(self, X):
        idx = 0
        predictions = []
        X = normalize_columns_fn(X)
        for _, row in X.iterrows():
            x = list(row)
            prediction = self.get_min_distance_class(x)
            predictions.append(prediction)
            idx += 1
        return predictions
            
    def compute_scores(self, predictions, Y):
        ground_truths = list(Y)
        accuracy = metrics.accuracy_score(ground_truths, predictions)
        F1 = metrics.f1_score(ground_truths, predictions)
        confusion_matrix = metrics.confusion_matrix(ground_truths, predictions)
        results = {'F1': F1, 'accuracy': accuracy, 'confusion_matrix': confusion_matrix}
        print(results)
        return results
