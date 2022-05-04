from sklearn import metrics
import pandas as pd
from data_loader import KFoldsGenerator, normalize_columns_fn


class ModelHandler(object):
    def __init__(self, model, data_loader, use_kfolds=True):
        self.model = model
        self.data_loader = data_loader
        self.use_kfolds = use_kfolds

    def compute_metrics(self, fitted_model, X, Y):
        pred = fitted_model.predict(X)
        f1_score = metrics.f1_score(Y, pred)
        accuracy_score = metrics.accuracy_score(Y, pred)
        confusion_matrix = metrics.confusion_matrix(Y, pred)
        print(f"F1 score {f1_score}")
        print(f"Accuracy {accuracy_score}")
        print(f"Confusion matrix\n {confusion_matrix}")
        results = {'F1': f1_score, 'accuracy': accuracy_score, 'confusion_matrix': confusion_matrix}
        return results

    def run_model(self, run_test=False):
        print("========================")
        train_X, train_Y, test_X, test_Y = self.data_loader.get_data()
        train_X = normalize_columns_fn(train_X)
        test_X = normalize_columns_fn(test_X)
        if self.use_kfolds:
            KFolds = KFoldsGenerator(self.data_loader.train_df)
            metrics_holder = []
            for fold in range(KFolds.k_folds):
                print(f"Fold {fold}")
                train_X, train_Y, val_X, val_Y = next(KFolds)
                train_X  = normalize_columns_fn(train_X)
                val_X = normalize_columns_fn(val_X)
                fitted_model = self.model.fit(train_X, train_Y)
                result_metrics = self.compute_metrics(fitted_model, val_X, val_Y)
                metrics_holder.append(result_metrics)
            metrics_df = pd.DataFrame(metrics_holder)
            avg_f1_score = metrics_df['F1'].mean()
            avg_accuracy = metrics_df['accuracy'].mean()
            print(f"Avg F1 cross val score {avg_f1_score}\nAvg accuracy cross val score {avg_accuracy}")
            print("========================")
        if run_test:
            print("Evaluating on test set")
            fitted_model = self.model.fit(train_X, train_Y)
            results = self.compute_metrics(fitted_model, test_X, test_Y)
            print("========================")
            return results
        return {'avg_f1_score': avg_f1_score, 'avg_accuracy': avg_accuracy}
