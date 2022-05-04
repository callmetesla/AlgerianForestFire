from data_loader import DataLoader, KFoldsGenerator, normalize_columns_fn
from generate_features import FeatureExpansion
from baseline_model import NearestMeans
from model_handler import ModelHandler

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

model_instances = {
    "KNN": KNeighborsClassifier(n_neighbors=24),
    "NB": GaussianNB(),
    "SVM": SVC(kernel="linear", random_state=42),
    "DecisionTree": tree.DecisionTreeClassifier(),
    "RandomForest": ensemble.RandomForestClassifier(max_depth=2, random_state=0),
    "MLP": MLPClassifier(random_state=1, max_iter=3600)
}

def train_model(model, X, Y):
    return model.fit(X, Y)


def compute_metrics(model, X, Y):
    pred = model.predict(X)
    f1_score = metrics.f1_score(Y, pred)
    accuracy_score = metrics.accuracy_score(Y, pred)
    print(f"F1 score {f1_score}")
    print(f"Accuracy {accuracy_score}")
    print(f"Confusion matrix\n {metrics.confusion_matrix(Y, pred)}")
    return f1_score, accuracy_score

def baseline_run(data_loader=None, run_test=False):
    if data_loader == None:
        data_loader = DataLoader(train_path, test_path)
    KFolds = KFoldsGenerator(data_loader.train_df)
    metrics_holder = []
    train_X, train_Y, test_X, test_Y = data_loader.get_data()
    train_X = normalize_columns_fn(train_X)
    test_X = normalize_columns_fn(test_X)
    for fold in range(KFolds.k_folds):
        print(f"Fold {fold}")
        train_X, train_Y, val_X, val_Y = next(KFolds)
        train_X = normalize_columns_fn(train_X)
        val_X = normalize_columns_fn(val_X)
        model = NearestMeans()
        model.fit(train_X, train_Y)
        predictions = model.predict(val_X)
        result_metrics = model.compute_scores(predictions, val_Y)
        metrics_holder.append(result_metrics)
    metrics_df = pd.DataFrame(metrics_holder)
    avg_f1_score = metrics_df['F1'].mean()
    avg_accuracy = metrics_df['accuracy'].mean()
    print(f"Avg F1 cross val score {avg_f1_score}\nAvg accuracy cross val score {avg_accuracy}")
    print("========================")
    if run_test:
        print("Evaluating on test set")
        model = NearestMeans()
        model.fit(train_X, train_Y)
        predictions = model.predict(test_X)
        model.compute_scores(predictions, test_Y)
        print("========================")
    return {'avg_accuracy': avg_accuracy, 'avg_f1_score': avg_f1_score, 'model': 'NearestMeans'}

def trivial_run(data_loader=None, run_test=False):
    if data_loader == None:
        data_loader = DataLoader(train_path, test_path)
    KFolds = KFoldsGenerator(data_loader.train_df)
    metrics_holder = []
    train_X, train_Y, test_X, test_Y = data_loader.get_data()
    train_X = normalize_columns_fn(train_X)
    test_X = normalize_columns_fn(test_X)
    for fold in range(KFolds.k_folds):
        print(f"Fold {fold}")
        train_X, train_Y, val_X, val_Y = next(KFolds)
        train_X = normalize_columns_fn(train_X)
        val_X = normalize_columns_fn(val_X)
        model = TrivialModel()
        model.fit(train_X, train_Y)
        predictions = model.predict(val_X)
        result_metrics = model.compute_scores(predictions, val_Y)
        metrics_holder.append(result_metrics)
    metrics_df = pd.DataFrame(metrics_holder)
    avg_f1_score = metrics_df['F1'].mean()
    avg_accuracy = metrics_df['accuracy'].mean()
    print(f"Avg F1 cross val score {avg_f1_score}\nAvg accuracy cross val score {avg_accuracy}")
    print("========================")
    if run_test:
        print("Evaluating on test set")
        model = TrivialModel()
        model.fit(train_X, train_Y)
        predictions = model.predict(test_X)
        model.compute_scores(predictions, test_Y)
        print("========================")
    cols = ','.join(data_loader.train_df.columns)
    return {'avg_accuracy': avg_accuracy, 'avg_f1_score': avg_f1_score, 'model': 'Trivial', 'cols': cols}


def mulitple_models_output():
    data_loader = DataLoader(train_path, test_path)
    for model_name, model_instance in model_instances.items():
        print("========================")
        print(f"Evaluating model {model_name}")
        m = ModelHandler(model_instance, data_loader)
        m.run_model()

def try_best_features(best_cols, window_size):
    best_cols_copy = best_cols
    best_cols.extend(['Classes', 'Date'])
    best_cols = set(best_cols)
    data_loader = DataLoader(train_path, test_path)
    train_df_new, test_df_new = FeatureExpansion(data_loader.train_df, data_loader.test_df).permute_all_features(window_size)
    data_loader.train_df = train_df_new
    data_loader.test_df = test_df_new
    data_loader.filter_columns(best_cols)
    experiment_holder = []
    experiment_holder.append(baseline_run(data_loader))
    experiment_holder.append(trivial_run(data_loader))
    for model_name, model_instance in model_instances.items():
        print("========================")
        print(f"Evaluating model {model_name}")
        m = ModelHandler(model_instance, data_loader)
        exp_out = m.run_model()
        exp_out['model'] = model_name
        exp_out['cols'] = ",".join(best_cols_copy)
        experiment_holder.append(exp_out)
    return pd.DataFrame(experiment_holder).sort_values('avg_accuracy', ascending=False)[['model', 'avg_accuracy', 'avg_f1_score', 'cols']]

def try_basic_run(outpath):
    data_loader = DataLoader(train_path, test_path)
    experiment_holder = []
    experiment_holder.append(baseline_run(data_loader))
    experiment_holder.append(trivial_run(data_loader))
    for model_name, model_instance in model_instances.items():
        print("========================")
        print(f"Evaluating model {model_name}")
        m = ModelHandler(model_instance, data_loader)
        exp_out = m.run_model()
        exp_out['cols'] = ",".join(data_loader.train_df.columns)
        exp_out['model'] = model_name
        experiment_holder.append(exp_out)
    return pd.DataFrame(experiment_holder).sort_values('avg_accuracy', ascending=False)[['model', 'avg_accuracy', 'avg_f1_score']].to_csv(f'{outpath}/basic.csv')

