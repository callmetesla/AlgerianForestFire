from data_loader import DataLoader, KFoldsGenerator
from baseline_model import NearestMeans
from trivial_model import TrivialModel
import pandas as pd
from generate_features import FeatureExpansion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


train_path = 'dataset/algerian_fires_train.csv'
test_path = 'dataset/algerian_fires_test.csv'


d = DataLoader(train_path, test_path)
f = FeatureExpansion(d.train_df, d.test_df)
train_df_new, test_df_new = f.permute_all_features(2)
d.train_df = train_df_new
d.test_df = test_df_new
best_cols = set(['BUI', 'BUI_max', 'DMC', 'FFMC', 'FFMC_max', 'ISI', 'ISI_median', 'Temperature', 'Ws_median', 'Classes', 'Date'])
d.filter_columns(best_cols)
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear']}
svc = SVC()
p={'C': [1, 10]}
KFolds = KFoldsGenerator(d.train_df)
cv = GridSearchCV(svc, p, cv=KFolds)
train_X, train_Y, test_X, test_Y = d.get_data()
cv.fit(train_X, train_Y)
#cv.fit()
print(cv.best_params_)
print(cv.best_estimator_)

#svc = SVC(C=0.1, kernel='linear', gamma=1)
#svc.fit(train_X, train_Y)
#print(svc.score(test_X, test_Y))
#f = FeatureExpansion(d.train_df, d.test_df)
#td, ttd = f.permute_all_features(7)
#k = KFoldsGenerator(d.train_df, 5)
#for _ in range(5):
#    print("---")
"""
train_X, train_Y, test_X, test_Y = d.get_data()
n = NearestMeans()
n.fit(train_X, train_Y)
predictions = n.predict(test_X)
n.compute_scores(predictions, test_Y)
#n.predict(test_X, test_Y)
"""
"""
trivial_model = TrivialModel()
trivial_model.fit(train_X, train_Y)
predictions = trivial_model.predict(test_X)
trivial_model.compute_scores(predictions, test_Y)
"""
