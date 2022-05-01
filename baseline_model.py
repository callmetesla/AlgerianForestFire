import pandas as pd
import numpy as np
from sklearn import metrics

class NearestMeans(object):
    def __init__(self):
        self.means = []
    
    def compute_means(self, train_X, train_Y):
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
        print(self.means)

    def compute_distance(self, x, y):
        return np.linalg.norm(x - y)

    def get_min_distance_class(self, x, num_classes=2):
        distances = []
        for i in range(num_classes):
            distances.append(np.linalg.norm(x-self.means[i]))
        print(distances)
        return np.argmin(distances)

    def predict(self, X, Y):
        idx = 0
        ground_truths = []
        predictions = []
        for _, row in X.iterrows():
            x = list(row)
            ground_truth = Y.iloc[idx]
            prediction = self.get_min_distance_class(x)
            ground_truths.append(ground_truth)
            predictions.append(prediction)
            idx += 1
        print(metrics.accuracy_score(ground_truths, predictions))
        print(metrics.f1_score(ground_truths, predictions))
        print(metrics.confusion_matrix(ground_truths, predictions))
            
