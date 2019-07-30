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


class GreedyMSEMinimizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def predict_is_imputed(self, X, step_size, extra_info=False):
        probs = self.model.probability_is_imputed(X)
        probs = probs.numpy()
        acc = []
        for i in tqdm.tqdm(range(0, probs.shape[0])):
            v = pd.Series(X[i, :])
            p = pd.Series(probs[i, :])
            result = minimize_mse_(v, p, step_size)
            result = {k: numpy2ri.ri2py(v) for v in result.items()}
            result = {k: v[0] if len(v) == 1 else v for k, v in result.items()}
            acc.append(result)
        df = pd.DataFrame(acc)
        yhat = np.array([e for e in df.predicted.values])
        if extra_info:
            return yhat, df
        else:
            return yhat

    def probability_is_imputed(self, X):
        # prob == 1.0 if we predict it, just for easy evaluation
        yhat = self.predict_is_imputed(self, X, extra_info=False)
        return yhat.astype(float)


class RepairImpute(object):
    def __init__(self, model):
        self.model = model.eval()

    def remove_imputed(self, ts, threshold=0.5):
        is_imp = self.model.predict_is_imputed(ts, threshold)
        is_imp = is_imp.numpy().astype(bool)
        ts[is_imp] = np.nan
        return ts

    def reimpute(self, ts, method, threshold=0.5):
        ts_with_missing = self.remove_imputed(ts, threshold)
        nrows = ts_with_missing.shape[0]
        filled = []
        for i in range(nrows):
            tsw = pd.Series(ts_with_missing[i, :])
            tsf = impute_missing_(tsw, method)
            filled.append(tsf)
        result = np.vstack(filled)
        return result
