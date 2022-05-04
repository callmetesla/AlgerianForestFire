from data_loader import DataLoader, normalize_columns_fn
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from generate_features import FeatureExpansion
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from experiments import try_best_features, try_basic_run
import matplotlib.pyplot as plt

ESTIMATOR = SVC(kernel="linear", random_state=42)
TRAIN_PATH = 'dataset/algerian_fires_train.csv'
TEST_PATH = 'dataset/algerian_fires_test.csv'
SAVE_DFS = False


def run_for_best_features(best_cols, window_size, method, outpath):
    df = try_best_features(best_cols, window_size)
    if SAVE_DFS:
        df.to_csv(f'{outpath}/{method}.csv')
    return {'method': method, 'df': df}


def print_best_cols(best_cols, method):
    best_cols = ','.join(sorted(best_cols))
    print(f"{method}: {best_cols}")

def get_train_data(data_loader):
    train_X, train_Y, _, _ = data_loader.get_data()
    train_X = normalize_columns_fn(train_X)
    return train_X, train_Y

def pearson_correlation(train_df_new, outpath, window_size):
    train_df_correlation = train_df_new.corr()
    best_cols = list(train_df_correlation['Classes'].sort_values(ascending=False)[0:10].index)
    print_best_cols(best_cols, 'Pearson')
    run_for_best_features(best_cols, window_size, 'Pearson', outpath)


def SFS(data_loader, outpath, window_size):
    train_X, train_Y = get_train_data(data_loader)
    sfs = SequentialFeatureSelector(ESTIMATOR, n_features_to_select=9)
    sfs.fit(train_X, train_Y)
    orig_cols = train_X.columns
    masks = sfs.get_support()
    best_cols = sorted(orig_cols[masks])
    print_best_cols(best_cols, 'SFS')
    run_for_best_features(best_cols, window_size, 'SFS', outpath)


def SFSBackward(data_loader, outpath, window_size):
    train_X, train_Y = get_train_data(data_loader)
    sfs = SequentialFeatureSelector(ESTIMATOR, n_features_to_select=9, direction='backward')
    sfs.fit(train_X, train_Y)
    orig_cols = train_X.columns
    masks = sfs.get_support()
    best_cols = sorted(orig_cols[masks])
    print_best_cols(best_cols, 'SFSBackward')
    run_for_best_features(best_cols, window_size, 'SFSBackward', outpath)


def RFES(data_loader, outpath, window_size):
    train_X, train_Y = get_train_data(data_loader)
    rfe = RFE(ESTIMATOR, n_features_to_select=9, step=1)
    train_X = normalize_columns_fn(train_X)
    rfe.fit(train_X, train_Y)
    orig_cols = train_X.columns
    masks = rfe.get_support()
    best_cols = sorted(orig_cols[masks])
    print_best_cols(best_cols, 'RFE')
    run_for_best_features(best_cols, window_size, 'RFE', outpath)


def UFS(data_loader, outpath, window_size):
    train_X, train_Y = get_train_data(data_loader)
    train_X = normalize_columns_fn(train_X)
    sel = SelectKBest(chi2, k=9)
    orig_cols = train_X.columns
    cols = sel.fit_transform(train_X, train_Y)
    masks = sel.get_support()
    best_cols = sorted(orig_cols[masks])
    print_best_cols(best_cols, 'UFE')
    run_for_best_features(best_cols, window_size, 'UFE', outpath)


def do_experiment(window_size):
    outpath = f'window_{window_size}_outs'
    data_loader = DataLoader(TRAIN_PATH, TEST_PATH)
    train_df_new, test_df_new = FeatureExpansion(data_loader.train_df, data_loader.test_df).permute_all_features(window_size)
    pearson_correlation(train_df_new, outpath, window_size)
    data_loader.train_df = train_df_new
    SFS(data_loader, outpath, window_size)
    SFSBackward(data_loader, outpath, window_size)
    UFS(data_loader, outpath, window_size)
    RFES(data_loader, outpath, window_size)
    try_basic_run(outpath)

do_experiment(window_size=2)
