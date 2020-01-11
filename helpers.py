import re
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from datetime import datetime
from sklearn.model_selection import cross_val_predict, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error


class Preprocessor:
    def __init__(self, train_data_path, test_data_path):
        """Initialization

        @param train_data_path
        @param test_data_path
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def _conv_numeric_feature(self, feature):
        feature = feature.replace(',', '')
        vals = [int(val) for val in re.findall('\d+', feature)]
        if len(vals) == 1:
            return vals[0] * 1000 if 'K' in feature else vals[0] 
        elif len(vals) == 2:
            if '.' in feature:
                value = vals[0] + 0.1 * vals[1]
                if 'K' in feature:
                    value = value * 1000
                return value
            return None
        else:
            return None

    def _extract_year_month(self, ts_str):
        ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        return ts.year, ts.month

    def _engineer_features(self, df):
        '''Process features '''
        # Covert "Likes" and "Popularity" to numeric data type
        df['Likes'] = df['Likes'].apply(lambda likes: self._conv_numeric_feature(likes))
        df['Popularity'] = df['Popularity'].apply(lambda popularity: self._conv_numeric_feature(popularity))

        # Extract release year & month from timestamp
        lambda_expr = lambda ts_str: pd.Series(self._extract_year_month(ts_str))
        df[['Release_Year', 'Release_Month']] = df['Timestamp'].apply(lambda_expr)

        # Map Genre
        group0 = ['indie', 'hiphoprap', 'classical', 'all-music', 'reggaeton',
                  'dubstep', 'rbsoul', 'drumbass', 'disco']
        group3 = ['alternativerock', 'country', 'metal', 'deephouse', 'trap',
                  'latin', 'ambient', 'pop', 'rock', 'folksingersongwriter']
        df.loc[df['Genre'].isin(group0), 'Categorical_Genre'] = 0
        df.loc[df['Genre'] == 'danceedm', 'Categorical_Genre'] = 1
        df.loc[df['Genre'] == 'electronic', 'Categorical_Genre'] = 2
        df.loc[df['Genre'].isin(group3), 'Categorical_Genre'] = 3

        # Map Release Year & Month
        df.loc[df['Release_Year'] < 2016, 'Categorical_Year'] = 0
        df.loc[df['Release_Year'].isin([2016, 2017]), 'Categorical_Year'] = 1
        df.loc[df['Release_Year'] > 2017, 'Categorical_Year'] = 2
        df.loc[df['Release_Month'].isin([12, 1, 2]), 'Categorical_Month'] = 0
        df.loc[df['Release_Month'].isin([3, 4, 5]), 'Categorical_Month'] = 1
        df.loc[df['Release_Month'].isin([6, 7, 8]), 'Categorical_Month'] = 2
        df.loc[df['Release_Month'].isin([9, 10, 11]), 'Categorical_Month'] = 3

        df['Categorical_Genre'] = df['Categorical_Genre'].astype(int)
        df['Categorical_Year'] = df['Categorical_Year'].astype(int)
        df['Categorical_Month'] = df['Categorical_Month'].astype(int)
        
        return df

    def _process_train_data(self, train_df):
        df = self._engineer_features(train_df)

        # Perform Box-Cox transformation on numeric features
        fitted_lambda = {}
        df['BC_Views'], fitted_lambda['Views'] = stats.boxcox(df['Views'] + 1)
        df['BC_Comments'], fitted_lambda['Comments'] = stats.boxcox(df['Comments'] + 1)
        df['BC_Likes'], fitted_lambda['Likes'] = stats.boxcox(df['Likes'] + 1)
        df['BC_Popularity'], fitted_lambda['Popularity'] = stats.boxcox(df['Popularity'] + 1)
        df['BC_Followers'], fitted_lambda['Followers'] = stats.boxcox(df['Followers'] + 1)
        print("fitted_lambda: ", fitted_lambda)

        # One-hot encode the categorical features
        encoded_df = pd.get_dummies(df, columns=['Categorical_Genre', 'Categorical_Year', 'Categorical_Month'])

        # Drop original features
        drop_cols = ['Name', 'Genre', 'Country', 'Song_Name', 'Timestamp', 'Views',
                     'Comments', 'Likes', 'Popularity', 'Followers', 'Release_Year', 'Release_Month']
        encoded_df = encoded_df.drop(drop_cols, axis=1)
        return encoded_df, fitted_lambda

    def _process_test_data(self, test_df, boxcox_lambdas):
        df = self._engineer_features(test_df)
        
        # Perform Box-Cox transformation with the lambda parameters
        # used on the training data
        df['BC_Comments'] = stats.boxcox(df['Comments'] + 1, boxcox_lambdas.get('Comments'))
        df['BC_Likes'] = stats.boxcox(df['Likes'] + 1, boxcox_lambdas.get('Likes'))
        df['BC_Popularity'] = stats.boxcox(df['Popularity'] + 1, boxcox_lambdas.get('Popularity'))
        df['BC_Followers'] = stats.boxcox(df['Followers'] + 1, boxcox_lambdas.get('Followers'))

        # One-hot encode the categorical features
        encoded_df = pd.get_dummies(df, columns=['Categorical_Genre', 'Categorical_Year', 'Categorical_Month'])

        # Drop original features
        drop_cols = ['Name', 'Genre', 'Country', 'Song_Name', 'Timestamp',
                     'Comments', 'Likes', 'Popularity', 'Followers', 'Release_Year', 'Release_Month']
        encoded_df = encoded_df.drop(drop_cols, axis=1)
        return encoded_df

    def run(self):
        print("Start processing training and test datasets ...")
        start_time = time.time()

        # Process training data
        train_df = pd.read_csv(self.train_data_path)
        print("Process training data: ", self.train_data_path)
        processed_train_df, boxcox_lambdas = self._process_train_data(train_df)
        print("Elapsed time: %s seconds" % round(time.time() - start_time, 4))
        print()

        # Process test data
        test_df = pd.read_csv(self.test_data_path)
        print("Process test data: ", self.test_data_path)
        processed_test_df = self._process_test_data(test_df, boxcox_lambdas)
        print("Elapsed time: %s seconds" % round(time.time() - start_time, 4))
        return processed_train_df, processed_test_df, boxcox_lambdas


class ModelWorker:
    def __init__(self, model, scores, boxcox_lambdas):
        """Initilization

        @param model
        @param scores   # scores = ['neg_mean_squared_error']
        @param param_grid

        """
        self.model = model
        self.scores = scores
        self.boxcox_lambdas = boxcox_lambdas

    def _evaluate(self, y_pred, y_true):
        '''Evaluate performance of classification
        '''
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print("RMSE: ", round(rmse, 6))
    
    def fit(self, X_train, y_train, features, nfolds=5):
        """Fit machine learning model and evaluate performance
        on the training dataset

        """
        y_pred = cross_val_predict(self.model, X_train[features], y_train, cv=nfolds)
        self._evaluate(y_pred, y_train)

    def tune(self, param_grid, X_train, y_train, nfolds, grid_search=True):
        start_time = time.time()
        best_score = {}
        for score in self.scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            if grid_search:
                clf = GridSearchCV(self.model, param_grid, cv=KFold(n_splits=nfolds),
                                   scoring='%s' % score)
            else:
                clf = RandomizedSearchCV(self.model, param_grid, cv=KFold(n_splits=nfolds),
                                         scoring='%s' % score, random_state=106)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            best_score[score] = max(means)
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                    % (mean, std * 2, params))
            print("Elapsed time: %s seconds" % round(time.time() - start_time, 4))
            print()
        
        return clf, best_score

    def predict(self, X_train, y_train, X_test, features):
        self.model.fit(X_train[features], y_train)
        print("Fitted model: ", self.model)
        X_test['BC_Views'] = self.model.predict(X_test[features])
        X_test['Views'] = inv_boxcox(X_test['BC_Views'], self.boxcox_lambdas.get('Views')) - 1
        return X_test