from argparse import ArgumentParser

import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import f1_score

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
        probs = self.probability_is_imputed(*args, **kwargs)
        threshold = kwargs.get("threshold", 0.0)
        yhat = probs > threshold
        return yhat.astype(bool)

    def probability_is_imputed(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")

    def __call__(self, *args, **kwargs):
        return self.probability_is_imputed(*args, **kwargs)


class GreedyMSEMinimizer(ModelWrapper):
    def __init__(self, model, step_size=0.1):
        super().__init__()
        self.model = model.eval()
        self.step_size = step_size

    def probability_is_imputed(
            self, X, step_size=None, extra_info=False, **kwargs
    ):
        # produces probabilities of 1 and 0 only, since makes decisions
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

    def probability_is_imputed(self, X, **kwargs):
        nrows = X.shape[0]
        results = []
        for i in tqdm.tqdm(range(0, nrows)):
            v = pd.Series(X[i, :])
            vhat = numpy2ri.ri2py(self.r_function(v))
            results.append(np.isnan(vhat))
        return np.array(results).astype(float)


class ManualBaseline(ModelWrapper):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.default_threshold = threshold

    def fit(self, X, y):
        y = y.flatten()
        best_f1 = None
        best_threshold = None

        for threshold in tqdm.tqdm(np.linspace(0, 1, n_steps + 1)):
            yhat = self.probability_is_imputed(X, threshold=threshold)
            yhat = yhat.flatten()
            f1 = f1_score(y, yhat)
            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        self.default_threshold = best_threshold
        return self

    def probability_is_imputed(self, X, threshold=None, **kwargs):
        zero_or_ones = (X == 0) | (X == 1)
        nrows = X.shape[0]
        nsteps = X.shape[1]
        preds = []
        if threshold is None:
            threshold = self.default_threshold
        for i in tqdm.tqdm(range(0, nrows)):
            row = X[i, :]
            row_indicator = np.repeat(False, nsteps)
            # repeated constants: possibly mean/mode
            vals, counts = np.unique(row, return_counts=True)
            if max(counts) > 1:
                rep_val = vals[np.argmax(counts)]
                row_indicator[row == rep_val] = True
            # repeated values in sequence, possible fwd/bwd (can only identify)
            # one otherwise end up labeling both
            row_0_to_n_minus_1 = row[:(nsteps - 1)]
            row_1_to_n = row[1:]
            possible_fwd = np.append(False, row_0_to_n_minus_1 == row_1_to_n)
            row_indicator |= possible_fwd
            # large difference in values
            abs_pct_diffs = np.abs(row_1_to_n - row_0_to_n_minus_1
                                   ) / np.abs(row_0_to_n_minus_1)
            possible_shift = np.append([False], abs_pct_diffs > threshold)
            row_indicator |= possible_shift
            preds.append(row_indicator)
        preds = np.array(preds)
        return preds.astype(float)


def get_tsoutliers_baseline():
    return RBaseline(ts_outliers_)


def get_tsclean_baseline():
    return RBaseline(forecast_tsclean_)


def get_manual_baseline(threshold=0.2):
    return ManualBaseline(threshold)
