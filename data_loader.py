import pandas as pd
import numpy as np


def normalize_columns_fn(df):
    for col in (df.columns):
        df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
    return df


class DataLoader(object):
    def __init__(self, train_csv_path, test_csv_path):
        self.train_df = self._load_data(train_csv_path)
        self.test_df = self._load_data(test_csv_path)

    def _load_data(self, path):
        return pd.read_csv(path)
    
    def filter_columns(self, columns):
        print(columns)
        self.train_df = self.train_df[columns]
        self.test_df = self.test_df[columns]

    def get_data(self):
        train_Y = self.train_df['Classes']
        train_X = self.train_df.drop(columns=['Date', 'Classes'])
        test_Y = self.test_df['Classes']
        test_X = self.test_df.drop(columns=['Date', 'Classes'])
        return train_X, train_Y, test_X, test_Y
    
class KFoldsGenerator(object):
    def __init__(self, train_df,  k_folds=5):
        self.train_df = train_df
        self.k_folds = k_folds
        self.folds = self._perform_k_folds()
        self.fold_counter = 0
    
    def _perform_k_folds(self):
        train_dates = self.train_df['Date'].unique()
        train_k_idxs = np.array_split(train_dates, self.k_folds)
        print(train_k_idxs)
        out_list = []
        for i in range(self.k_folds):
            inner_list = []
            for date in train_k_idxs[i]:
                selected_rows = self.train_df[self.train_df['Date'] == date]
                inner_list.append(selected_rows)
            out_list.append(pd.concat(inner_list))
        return out_list
    
    def __next__(self):
        val_data = self.folds[self.fold_counter]
        train_data = self.folds[:self.fold_counter] + self.folds[self.fold_counter + 1:]
        train_data = pd.concat(train_data)
        train_Y = train_data['Classes']
        train_X = train_data.drop(columns=['Date', 'Classes'])
        val_Y = val_data['Classes']
        val_X = val_data.drop(columns=['Date', 'Classes'])
        self.fold_counter += 1
        return train_X, train_Y, val_X, val_Y
