from data_loader import DataLoader, KFoldsGenerator
from baseline_model import NearestMeans
import pandas as pd
from generate_features import FeatureExpansion


train_path = 'dataset/algerian_fires_train.csv'
test_path = 'dataset/algerian_fires_test.csv'


d = DataLoader(train_path, test_path)

#f = FeatureExpansion(d.train_df, d.test_df)
#td, ttd = f.permute_all_features(7)
#k = KFoldsGenerator(d.train_df, 5)
#for _ in range(5):
#    print("---")

train_X, train_Y, test_X, test_Y = d.get_data()

n = NearestMeans()
n.fit(train_X, train_Y)
n.predict(train_X,train_Y)
#n.predict(test_X, test_Y)
