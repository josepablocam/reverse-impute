from argparse import ArgumentParser

import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import numpy as np
import torch
import tqdm

pandas2ri.activate()
rinterpreter = robjects.r
rinterpreter.source("timeseries.R")

impute_missing_ = rinterpreter("impute_missing")
minimize_mse_ = rinterpreter("minimize_mse")


class GreedyMSEMinimizer(object):
    def __init__(self, model):
        self.model = model.eval()

    def predict_is_imputed(self, X):
        probs = self.model.probability_is_imputed(X)
        probs = probs.numpy()
        acc = []
        for i in tqdm.tqdm(range(0, probs.shape[0])):
            result = minimize_mse_(probs[i, :])
            result = zip(result.names, result.values)
            acc.append(result)
        df = pd.DataFrame(acc)
        return df

    def probability_is_imputed(self, X):
        # prob == 1.0 if we predict it, just for easy evaluation
        df = self.predict_is_imputed(self, X)
        return df.predicted.values.astype(float)


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
