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
