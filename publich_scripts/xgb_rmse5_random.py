import time

start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline  # model_selection
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.metrics import edit_distance
from nltk.stem.porter import *
from xgbRegression import XGBoostRegressor
from sklearn import pipeline, grid_search


stemmer = PorterStemmer()
import re
import random

random.seed(2016)
import xgboost as xgb
import sqlite3

df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
num_train = df_train.shape[0]
df_all = pd.read_csv('df_all_rmse5_4.csv', encoding="ISO-8859-1", index_col=0)
train = df_all.iloc[:num_train]
test = df_all.iloc[num_train:]
print test.head()
id_test = test['id']
print id_test.head()
# balance train
c = sqlite3.connect(':memory:')
# c = sqlite3.connect('temp.db')
train.to_sql('t', c)
train = pd.read_sql('select * from t order by relevance desc, product_uid asc', c, index_col=['index'])
df_even = train.iloc[::2]  # even
df_odd = train.iloc[1::2]  # odd
train = pd.concat((df_even, df_odd), axis=0, ignore_index=True)
train = train.reset_index(drop=True)
y_train = train['relevance']
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time) / 60), 2))


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_


RMSE = make_scorer(fmean_squared_error, greater_is_better=False)


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        # d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand','search_and_prod_info','len_of_query', 'len_of_brand', 'query_in_description', 'word_in_brand', 'ratio_brand', 'ft04', 'ft05', 'ftx05', 'ftz04', 'ftz05', 'sft05', 'sftx05', 'sftz05','ft06', 'ft07', 'ft08', 'ft09', 'ft10', 'ft11', 'ft12', 'ft13', 'ft14', 'ftx06', 'ftx07', 'ftx08', 'ftx09', 'ftx10', 'ftx11', 'ftx12', 'ftx13', 'ftx14', 'ftz06', 'ftz07', 'ftz08', 'ftz09', 'ftz10', 'ftz11', 'ftz12', 'ftz13', 'ftz14', 'sft06', 'sft07', 'sft08', 'sft09', 'sft10', 'sft11', 'sft12', 'sft13', 'sft14', 'sft101', 'sft102', 'sft103', 'sft104', 'sft105', 'sft106', 'sft107', 'sft108', 'sft109', 'sft110', 'sft111', 'sft112', 'sft113', 'sft114', 'sftx06', 'sftx07', 'sftx08', 'sftx09', 'sftx10', 'sftx11', 'sftx12', 'sftx13', 'sftx14', 'sftx101', 'sftx102', 'sftx103', 'sftx104', 'sftx105', 'sftx106', 'sftx107', 'sftx108', 'sftx109', 'sftx110', 'sftx111', 'sftx112', 'sftx113', 'sftx114', 'sftz06', 'sftz07', 'sftz08', 'sftz09', 'sftz10', 'sftz11', 'sftz12', 'sftz13', 'sftz14', 'sftz101', 'sftz102', 'sftz103', 'sftz104', 'sftz105', 'sftz106', 'sftz107', 'sftz108', 'sftz109', 'sftz110', 'sftz111', 'sftz112', 'sftz113', 'sftz114']
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description', 'search_and_prod_info',
                       'product_info', 'attr', 'brand', 'len_of_query', 'len_of_brand', 'query_in_description',
                       'word_in_brand', 'ratio_brand', 'ft04', 'ft05', 'ftx05', 'ftz04', 'ftz05', 'sft05', 'sftx05',
                       'sftz05', 'ft06', 'ft07', 'ft08', 'ft09', 'ft10', 'ft11', 'ft12', 'ft13', 'ft14', 'ftx06',
                       'ftx07', 'ftx08', 'ftx09', 'ftx10', 'ftx11', 'ftx12', 'ftx13', 'ftx14', 'ftz06', 'ftz07',
                       'ftz08', 'ftz09', 'ftz10', 'ftz11', 'ftz12', 'ftz13', 'ftz14', 'sft06', 'sft07', 'sft08',
                       'sft09', 'sft10', 'sft11', 'sft12', 'sft13', 'sft14', 'sft101', 'sft102', 'sft103', 'sft104',
                       'sft105', 'sft106', 'sft107', 'sft108', 'sft109', 'sft110', 'sft111', 'sft112', 'sft113',
                       'sft114', 'sftx06', 'sftx07', 'sftx08', 'sftx09', 'sftx10', 'sftx11', 'sftx12', 'sftx13',
                       'sftx14', 'sftx101', 'sftx102', 'sftx103', 'sftx104', 'sftx105', 'sftx106', 'sftx107', 'sftx108',
                       'sftx109', 'sftx110', 'sftx111', 'sftx112', 'sftx113', 'sftx114', 'sftz06', 'sftz07', 'sftz08',
                       'sftz09', 'sftz10', 'sftz11', 'sftz12', 'sftz13', 'sftz14', 'sftz101', 'sftz102', 'sftz103',
                       'sftz104', 'sftz105', 'sftz106', 'sftz107', 'sftz108', 'sftz109', 'sftz110', 'sftz111',
                       'sftz112', 'sftz113', 'sftz114']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        print 'tag1'
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        print 'tag2'
        return data_dict[self.key].apply(unicode)



xgb_model = XGBoostRegressor(silent = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tsvd = TruncatedSVD(n_components=50, random_state=2016)
tnmf = NMF(n_components=50, random_state=2016)
clf = pipeline.Pipeline([
    ('union', FeatureUnion(
            transformer_list=[
                ('cst', cust_regression_vals()),
                ('txt1', pipeline.Pipeline(
                        [('s2', cust_txt_col(key='search_and_prod_info')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                ('txt2', pipeline.Pipeline(
                        [('s3', cust_txt_col(key='search_and_prod_info')), ('tfidf2', tfidf), ('tnmf', tnmf)])),
                ('txt3', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf3', tfidf)])),
                ('txt4', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                ('txt5', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf5', tfidf), ('tsvd5', tsvd)])),
                ('txt6', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf6', tfidf), ('tsvd6', tsvd)]))
            ],
            transformer_weights={
                'cst': 1.0,
                'txt1': 1.0,
                'txt2': 1.0,
                'txt3':1.0,
                'txt4': 1.0,
                'txt5': 1.0,
                'txt6': 1.0
            },
            n_jobs = 1
    )),
    ('xgbr', xgb_model)])
param_grid ={'xgbr__silent': [1], 'xgbr__nthread': [3], 'xgbr__eval_metric': ['rmse'], 'xgbr__eta': [0.01,0.02,0.03,0.04,0.05, 0.06],
             'xgbr__max_depth': [5,6,7,8,9,10], 'xgbr__num_round': [600, 700, 800, 900, 1000, 1200], 'xgbr__fit_const': [0.4,0.5,0.6,0.7],
             'xgbr__subsample': [0.5,0.66,0.75,0.8,0.9],'xgbr__objective':['reg:linear'],'xgbr__gamma':[0,1,5,10,100],
             'xgbr__min_child_weight':[1,3,6,10],'xgbr__colsample_bytree':[0.4,0.5,0.6,0.7]}
model = grid_search.RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_jobs = 1, cv = 3,n_iter=30, verbose = 0, scoring=RMSE)

model.fit(train, y_train.values)
print("----------------------------------")
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
print(model.best_score_ + 0.452493857808)
print("----------------------------------")
for i in range(len(model.grid_scores_)):
    print(model.grid_scores_[i][0], model.grid_scores_[i][1])
    print(model.grid_scores_[i][2])
    print("----------------------------------")

y_pred = model.predict(test)

min_y_pred = min(y_pred)
max_y_pred = max(y_pred)
min_y_train = min(y_train.values)
max_y_train = max(y_train.values)
print(min_y_pred, max_y_pred, min_y_train, max_y_train)
for i in range(len(y_pred)):
    if y_pred[i] < 1.0:
        y_pred[i] = 1.0
    if y_pred[i] > 3.0:
        y_pred[i] = 3.0
        # y_pred[i] = min_y_train + (((y_pred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('single_0414_rmse5_iter30_1.csv', index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time) / 60), 2))