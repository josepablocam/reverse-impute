from argparse import ArgumentParser

import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
import numpy as np
import torch
import torch.nn as nn
import tqdm

pandas2ri.activate()
rinterpreter = robjects.r
rinterpreter.source("timeseries.R")

impute_missing_ = rinterpreter("impute_missing")
minimize_mse_ = rinterpreter("minimize_mse")
forecast_tsclean_ = rinterpreter("forecast_tsclean")
ts_outliers_ = rinterpreter("tsoutliers_tsoutliers")


class ModelWrapper(object):
    def to(self, x):
        return self

    def eval(self, x):
        return self

    def predict_is_imputed(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")

    def probability_is_imputed(self, *args, **kwargs):
        yhat = self.predict_is_imputed(*args, **kwargs)
        return yhat.astype(float)

    def __call__(self, *args, **kwargs):
        return self.probability_is_imputed(*args, **kwargs)


class GreedyMSEMinimizer(ModelWrapper):
    def __init__(self, model, step_size=0.1):
        super().__init__()
        self.model = model.eval()
        self.step_size = step_size

    def predict_is_imputed(self, X, step_size=None, extra_info=False):
        if step_size is None:
            step_size = self.step_size
        probs = self.model.probability_is_imputed(X)
        probs = probs.numpy()
        acc = []
        for i in tqdm.tqdm(range(0, probs.shape[0])):
            v = pd.Series(X[i, :])
            p = pd.Series(probs[i, :])
            result = minimize_mse_(v, p, step_size)
            result = {k: numpy2ri.ri2py(v) for k, v in result.items()}
            result = {k: v[0] if len(v) == 1 else v for k, v in result.items()}
            acc.append(result)
        df = pd.DataFrame(acc)
        yhat = np.array([e for e in df.predicted.values])
        if extra_info:
            return yhat, df
        else:
            return yhat

class RBaseline(ModelWrapper):
    def __init__(self, r_function):
        super().__init__()
        self.r_function = r_function

    def predict_is_imputed(self, X):
        nrows = X.shape[0]
        results = []
        for i in tqdm.tqdm(range(0, nrows)):
            v = pd.Series(X[i, :])
            vhat = numpy2ri.ri2py(self.r_function(v))
            results.append(np.isnan(vhat))
        return np.array(results).astype(bool)


def get_tsoutliers_baseline():
    return RBaseline(ts_outliers_)

def get_tsclean_baseline():
    return RBaseline(forecast_tsclean_)


#
# class RepairImpute(object):
#     def __init__(self, model):
#         self.model = model.eval()
#
#     def remove_imputed(self, ts, threshold=0.5):
#         is_imp = self.model.predict_is_imputed(ts, threshold)
#         is_imp = is_imp.numpy().astype(bool)
#         ts[is_imp] = np.nan
#         return ts
#
#     def reimpute(self, ts, method, threshold=0.5):
#         ts_with_missing = self.remove_imputed(ts, threshold)
#         nrows = ts_with_missing.shape[0]
#         filled = []
#         for i in range(nrows):
#             tsw = pd.Series(ts_with_missing[i, :])
#             tsf = impute_missing_(tsw, method)
#             filled.append(tsf)
#         result = np.vstack(filled)
#         return result
