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
        ts_tensor = torch.tensor(ts).to(torch.float32).reshape(1, -1)
        with torch.no_grad():
            scores = self.model(ts_tensor)
        pred_is_imp = torch.sigmoid(scores) > threshold
        pred_is_imp = pred_is_imp.numpy().flatten()
        return pred_is_imp.astype(bool)

    def remove_imputed(self, ts, threshold=0.5):
        ts = ts.copy()
        is_imp = self.predict_is_imputed(ts, threshold)
        ts[is_imp] = np.nan
        return ts

    def reimpute(self, ts, method, threshold=0.5):
        ts_with_missing = self.remove_imputed(ts, threshold)
        ts_with_missing = pd.Series(ts_with_missing)
        filled = impute_missing_(ts_with_missing, method)
        return np.array(filled)
