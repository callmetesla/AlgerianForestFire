from data_loader import DataLoader, KFoldsGenerator
import numpy as np
import copy
from sklearn import metrics, preprocessing
import pandas as pd
from generate_features import FeatureExpansion


from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


train_path = 'dataset/algerian_fires_train.csv'
test_path = 'dataset/algerian_fires_test.csv'
model_instances = {
    "KNN": KNeighborsClassifier(n_neighbors=24),
    "NB": GaussianNB(),
    "SVM": SVC(kernel="linear", random_state=42)
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

def mulitple_models_output():
    data_loader = DataLoader(train_path, test_path)
    train_X, train_Y, test_X, test_Y = data_loader.get_data()
    min_max_scaler = preprocessing.MinMaxScaler()
    train_X = min_max_scaler.fit_transform(train_X)
    test_X = min_max_scaler.fit_transform(test_X)
    for model_name, model_instance in model_instances.items():
        print("========================")
        print(f"Evaluating model {model_name}")
        KFolds = KFoldsGenerator(data_loader.train_df)
        metrics_tracker = {'F1': [], 'Accuracy': []}
        for fold in range(KFolds.k_folds):
            model_copy = copy.deepcopy(model_instance)
            print(f"Fold {fold}")
            train_X, train_Y, val_X, val_Y = next(KFolds)
            train_X = min_max_scaler.fit_transform(train_X)
            val_X = min_max_scaler.fit_transform(val_X)
            fitted_model = train_model(model_copy, train_X, train_Y)
            f1, accuracy = compute_metrics(fitted_model, val_X, val_Y)
            metrics_tracker['F1'].append(f1)
            metrics_tracker['Accuracy'].append(accuracy)
            del model_copy
        avg_f1_score = np.array(metrics_tracker['F1']).mean()
        avg_accuracy = np.array(metrics_tracker['Accuracy']).mean()
        print(f"Avg F1 cross val score {avg_f1_score}\n Avg accuracy cross val score {avg_accuracy}")
        print("========================")
        print("Evaluating on test set")
        fitted_model = train_model(model_instance, train_X, train_Y)
        compute_metrics(fitted_model, test_X, test_Y)
        print("========================")

mulitple_models_output()
