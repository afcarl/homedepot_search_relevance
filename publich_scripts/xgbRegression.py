'''
Coding Just for Fun
Created by burness on 16/2/18.
'''

import xgboost as xgb
import numpy as np
from ml_metrics import rmse,auc
from sklearn.base import BaseEstimator, TransformerMixin
class XGBoostRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, **params):
        self.clf = None
        self.params = params
        self.params.update({'objective': 'reg:linear'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = self.params['num_round']
        dtrain = xgb.DMatrix(X, label=y)
        # params = self.params
        if X.shape[1]==1:
            self.params.update({'colsample_bytree': 1.0})
        self.clf = xgb.train(
                self.params,
                dtrain,
                num_boost_round=num_boost_round,
        )
        self.fscore = self.clf.get_fscore()

        bb = {}

        for ftemp, vtemp in self.fscore.items():
            bb[ftemp] = vtemp

        i = 0
        cc = np.zeros(dtrain.num_col())
        for feature, value in  bb.items():
            cc[i] = value
            i+=1
        self.coef_= cc

    def predict(self, X):
        dX = xgb.DMatrix(X)
        y = self.clf.predict(dX)
        return y

    def set_params(self, **params):
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self

    def get_params(self, deep=False):
        params = self.params
        return params