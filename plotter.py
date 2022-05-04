import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import array
import seaborn as sn

x_ticks = ['Trivial', 'KNN',  'NearestMeans', 'NB',  'RandomForest', 'SVM', 'MLP', 'DecisionTree']


gg = {'KNN': {'F1': 0.8163265306122449, 'accuracy': 0.85, 'confusion_matrix': array([[31,  6],
       [ 3, 20]])}, 'NB': {'F1': 0.8163265306122449, 'accuracy': 0.85, 'confusion_matrix': array([[31,  6],
       [ 3, 20]])}, 'SVM': {'F1': 0.8979591836734695, 'accuracy': 0.9166666666666666, 'confusion_matrix': array([[33,  4],
       [ 1, 22]])}, 'DecisionTree': {'F1': 0.7586206896551724, 'accuracy': 0.7666666666666667, 'confusion_matrix': array([[24, 13],
       [ 1, 22]])}, 'RandomForest': {'F1': 0.7843137254901961, 'accuracy': 0.8166666666666667, 'confusion_matrix': array([[29,  8],
       [ 3, 20]])}, 'MLP': {'F1': 0.84, 'accuracy': 0.8666666666666667, 'confusion_matrix': array([[31,  6],
       [ 2, 21]])}}
def load_df():
    dfhold = {}
    for csvf in os.listdir('window_2_outs'):
        fp = os.path.join('window_2_outs', csvf)
        df = pd.read_csv(fp)
        dfhold[csvf[:-4]] = df
    return dfhold

def collate():
    dfhold = load_df()
    avg_acc_modl = {}
    avg_f1_modl = {}
    methods = ['Pearson', 'SFS_Backward', 'without_feature_sel', 'SFS', 'UFS', 'RFE']
    for method in methods:
        avg_acc_modl[method] = []
        avg_f1_modl[method] = []
        for model in x_ticks:
            mdf = dfhold[method]
            selrows = mdf[mdf['model']==model]
            avg_acc_modl[method].append(float(selrows['avg_accuracy']))
            avg_f1_modl[method].append(float(selrows['avg_f1_score']))
    plot_curves(avg_acc_modl, avg_f1_modl) 

def plot_curves(accd, f1d):
    x = list(range(1, 9))
    f, ax = plt.subplots()
    for method, y in f1d.items():
        ax.plot(x_ticks, y, label=method)
        #ax.set_xticklabels(x_ticks)
        plt.legend()
    plt.xlabel('Model')
    plt.ylabel('Avg cross validation f1 score')
    plt.title('Rolling average window size 2')
    plt.show()
#plot_curves()


models = list(gg.keys())

for model in models:
    metrics = gg[model]
    df = pd.DataFrame(metrics['confusion_matrix'])
    sn.heatmap(df, annot=True, annot_kws={"size": 16})
    plt.show()
