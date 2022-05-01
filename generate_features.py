import pandas  as pd
import numpy as np
import copy

DEFAULT_COLS = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']

class FeatureExpansion(object):
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
    
    def compute_moving_averages(self, df, columns, operation, window_size):
        out_list = []
        original_columns = df.columns[1:] # Without date col

        # Merge duplicate rows in list values to enable rolling average
        for date in df['Date'].unique():
            series = dict((col, []) for col in original_columns)
            # Select rows by date
            date_df = df[df['Date']==date]
            for index, row in date_df.iterrows():
                for col in original_columns:
                    series[col].append(row[col])
            series['Date'] = date
            out_list.append(series)

        dates_list = pd.DataFrame(out_list)['Date']
        output_df = pd.DataFrame(out_list).drop(columns=['Date'])
        if operation == 'max':
            output_df = output_df[columns].applymap(lambda x: max(x)).shift(1).rolling(window=window_size).max()
        elif operation == 'min':
            output_df = output_df[columns].applymap(lambda x: min(x)).shift(1).rolling(window=window_size).min()
        elif operation == 'avg':
            output_df = output_df[columns].applymap(lambda x: np.mean(x)).shift(1).rolling(window=window_size).mean()
        elif operation == 'median':
            output_df = output_df[columns].applymap(lambda x: np.median(x)).shift(1).rolling(window=window_size).median()
        output_df['Date'] = dates_list
        #  Drop last few window_size rows to prevent pollution with test data
        return output_df


    def merge_moving_averages(self, original_df, moving_average_df, operation):
        out_df = pd.DataFrame()
        moving_average_df = moving_average_df.dropna()
        for i, row in moving_average_df.iterrows():
            selected_rows = original_df[original_df['Date'] == row['Date']]
            indexes = list(selected_rows.index)
            for indx in indexes:
                original_row = original_df.iloc[indx]
                for col in row.index[:-1]:
                    original_row[f"{col}_{operation}"] = row[col]
                out_df = out_df.append(original_row)
        out_df = out_df.drop(columns=original_df.columns)
        return out_df 

    def permute_all_features(self, window_size):
        operations = ['max', 'min', 'avg', 'median']
        train_df_out = self.train_df.copy()
        test_df_out = self.test_df.copy()
        train_df_copy = self.train_df.copy()
        test_df_copy = self.test_df.copy()
        for operation in operations:
            train_moving_averages = self.compute_moving_averages(train_df_copy, DEFAULT_COLS, operation, window_size)
            train_moving_averages.drop(train_moving_averages.tail(window_size).index, inplace=True)
            train_merged_df = self.merge_moving_averages(train_df_copy, train_moving_averages, operation)
            test_df_expanded = pd.concat([train_df_copy.tail(window_size * 2), test_df_copy])
            test_moving_averages = self.compute_moving_averages(test_df_expanded, DEFAULT_COLS, operation, window_size)
            test_merged_df = self.merge_moving_averages(test_df_copy, test_moving_averages, operation)
            train_df_out = train_df_out.join(train_merged_df).dropna()
            test_df_out = test_df_out.join(test_merged_df).dropna()
        return train_df_out, test_df_out
