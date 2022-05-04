from sklearn.svm import SVC
from model_handler import ModelHandler
from data_loader import DataLoader
from generate_features import FeatureExpansion
from sklearn.decomposition import PCA
from data_loader import DataLoader, KFoldsGenerator, normalize_columns_fn
from generate_features import FeatureExpansion
from baseline_model import NearestMeans
from model_handler import ModelHandler
import seaborn as sns

import numpy as np
import copy
from sklearn import metrics, preprocessing, tree, ensemble
from sklearn.neural_network import MLPClassifier
import pandas as pd
from generate_features import FeatureExpansion
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from trivial_model import TrivialModel



train_path = 'dataset/algerian_fires_train.csv'
test_path = 'dataset/algerian_fires_test.csv'
final_model = SVC(kernel='linear', random_state=42)

model_instances = {
    "KNN": KNeighborsClassifier(n_neighbors=24),
    "NB": GaussianNB(),
    "SVM": SVC(kernel="linear", random_state=42),
    "DecisionTree": tree.DecisionTreeClassifier(),
    "RandomForest": ensemble.RandomForestClassifier(max_depth=2, random_state=0),
    "MLP": MLPClassifier(random_state=1, max_iter=3600)
}



def final_dataset():
    data_loader = DataLoader(train_path, test_path)
    window_size = 2
    train_df_new, test_df_new = FeatureExpansion(data_loader.train_df, data_loader.test_df).permute_all_features(window_size)
    data_loader.train_df = train_df_new
    data_loader.test_df = test_df_new
    best_cols = ["BUI", "BUI_max", "DMC", "FFMC", "FFMC_max", "ISI", "ISI_median", "Temperature", "Ws_median", "Classes", "Date"]
    data_loader.filter_columns(best_cols)
    return data_loader


def final_run():
    data_loader = final_dataset()
    model_handler = ModelHandler(final_model, data_loader)
    model_handler.run_model(run_test=True)

def run_models_final():
    modouts = {}
    for model_name, model_instance in model_instances.items():
        data_loader = final_dataset()
        model_handler = ModelHandler(model_instance, data_loader)
        results = model_handler.run_model(run_test=True)
        modouts[model_name] = results
    print(modouts)
#final_run()
run_models_final()
