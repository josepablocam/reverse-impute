from argparse import ArgumentParser

import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import numpy as np
import torch

pandas2ri.activate()
rinterpreter = robjects.r
rinterpreter.source("timeseries.R")

impute_missing_ = rinterpreter("impute_missing")


class RepairImpute(object):
    def __init__(self, model):
        self.model = model.eval()

    def predict_is_imputed(self, ts, threshold=0.5):
        flatten = False
        if len(ts.shape) == 1:
            ts = ts.reshape(1, -1)
            flatten = True
        ts_tensor = torch.tensor(ts).to(torch.float32)
        with torch.no_grad():
            scores = self.model(ts_tensor)
        pred_is_imp = torch.sigmoid(scores) > threshold
        flag = pred_is_imp.numpy().astype(bool)
        if flatten:
            return flag.flatten()

    def remove_imputed(self, ts, threshold=0.5):
        flatten = False
        if len(ts.shape) == 1:
            ts = ts.reshape(1, -1)
            flatten = True
        is_imp = self.predict_is_imputed(ts, threshold)
        ts[is_imp] = np.nan
        if flatten:
            ts = ts.flatten()
        return ts

    def reimpute(self, ts, method, threshold=0.5):
        flatten = False
        if len(ts.shape) == 1:
            flatten = True
            ts = ts.reshape(1, -1)
        ts_with_missing = self.remove_imputed(ts, threshold)
        nrows = ts_with_missing.shape[0]
        filled = []
        for i in range(nrows):
            tsw = pd.Series(ts_with_missing[i, :])
            tsf = impute_missing_(tsw, method)
            filled.append(tsf)
        result = np.vstack(filled)
        if flatten:
            return result.flatten()
