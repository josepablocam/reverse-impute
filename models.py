import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from sklearn.metrics import f1_score

pandas2ri.activate()
rinterpreter = robjects.r
rinterpreter.source("timeseries.R")

impute_missing_ = rinterpreter("impute_missing")
minimize_mse_ = rinterpreter("minimize_mse")
forecast_tsclean_ = rinterpreter("forecast_tsclean")
ts_outliers_ = rinterpreter("tsoutliers_tsoutliers")


# version 1
# take final bilstm hidden state
# and create entries of form
# (H, value)
# these get fed into a deep feed forward network
# to then predict missing or not for this
class MissingPredictor(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers=3,
            hidden_size=50,
            activation="relu",
            dropout=0.5,
    ):
        super().__init__()

        hidden = []
        activ = {"relu": nn.ReLU}[activation]
        input_layer = [nn.Linear(input_size, hidden_size), activ()]
        output_layer = [nn.Linear(hidden_size, 1)]

        hidden_layers = []
        for _ in range(num_layers):
            l = nn.Linear(hidden_size, hidden_size)
            hidden_layers.append(l)
            hidden_layers.append(activ())
            if dropout is not None:
                hidden_layers.append(nn.Dropout(p=dropout))

        layers = []
        layers.extend(input_layer)
        layers.extend(hidden_layers)
        layers.extend(output_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, batch):
        # batch shape: (num ts, num steps, dim(context) + 1)
        # return shape: (num ts, num steps)
        return self.layers(batch).squeeze(2)


class AttentionBasedSummarizer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # add one in size, since we append index of timestep
        self.w = nn.Linear(hidden_size + 1, 1)

    def repeat_hidden_state_matrix(self, H):
        # H shape: (num ts, num steps, bilstm hidden size)
        # repeat same set of hidden vectors for each time step
        # H_ext shape: (num ts, num steps, num steps, bilstm hidden size)
        num_obs, num_steps = H.shape[:2]
        H_ext = H.unsqueeze(1).repeat(1, num_steps, 1, 1)
        return H_ext

    def compute_attention_weights(self, H_ext):
        # H_ext shape: (num ts, num steps, num steps, bilstm hidden size)
        # alpha shape: (num ts, num steps, num_steps, 1)
        # append an index indicating what time step to each
        num_obs, num_steps = H_ext.shape[:2]
        ixs = torch.arange(0, num_steps, dtype=torch.float32).to(H_ext.device)
        ixs = ixs.reshape(-1, 1).repeat(1, num_steps).unsqueeze(-1)
        ixs = ixs.unsqueeze(0).repeat(num_obs, 1, 1, 1)
        # append time step index to end of each H entry
        H_ext_with_ix = torch.cat((H_ext, ixs), dim=-1)
        alpha = torch.softmax(self.w(H_ext_with_ix), dim=2)
        return alpha

    def summarize(self, H):
        # H shape: (num ts, num steps, bilstm hidden size)
        # output shape: (num ts, num steps, bilstm hidden size)
        H_ext = self.repeat_hidden_state_matrix(H)
        alpha = self.compute_attention_weights(H_ext)
        weighted_H_ext = (alpha * H_ext).sum(dim=2)
        return weighted_H_ext


class SequenceEncoder(nn.Module):
    def __init__(self, hidden_size, encoder_type="bilstm", attention=True):
        super().__init__()

        if encoder_type == "bilstm":
            self.encoder = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = None
            if attention:
                self.attention = AttentionBasedSummarizer(hidden_size * 2)
        else:
            raise Exception("Unknown model type: {}".format(encoder_type))

    def zscore(self, batch):
        # batch shape: (num_ts, num steps, 1)
        means = batch.mean(dim=1).unsqueeze(2)
        sd = batch.std(dim=1).unsqueeze(2)
        return (batch - means) / sd

    def lagged_diff(self, batch, n_lag):
        n_ts = batch.shape[0]
        n_time = batch.shape[1]
        diffed_ts = batch[:, n_lag:] - batch[:, :(n_time - n_lag)]
        # first n differences are zero by definition
        zeros = torch.zeros((n_ts, n_lag, 1)).to(torch.float32).to(
            batch.device
        )
        return torch.cat((zeros, diffed_ts), dim=1)

    def forward(self, batch):
        # batch shape: (num ts, num steps, 1)
        # output shape: (num ts, num steps, num_dirs * hidden_size + 1)
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(2)
        batch = self.zscore(batch)
        batch = self.lagged_diff(batch, n_lag=1)
        H, (hn, cn) = self.encoder(batch)
        # append to each num steps the observed value as well
        # technically already captured by hidden state, but still
        # adding
        if self.attention is not None:
            H = self.attention.summarize(H)
        return torch.cat((H, batch), dim=2)


class ReverseImputer(nn.Module):
    def __init__(self, enc_hidden_size, pred_hidden_size, attention=False):
        super().__init__()
        self.encoder = SequenceEncoder(
            enc_hidden_size,
            encoder_type="bilstm",
            attention=attention,
        )
        self.predictor = MissingPredictor(
            input_size=enc_hidden_size * 2 + 1,
            hidden_size=pred_hidden_size,
            activation="relu",
        )
        self.loss_fun = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        return self.predictor(self.encoder(batch))

    def compute_loss(self, batch_x, batch_y):
        pred_y = self.forward(batch_x)
        return self.loss_fun(pred_y, batch_y)

    def get_device(self):
        return next(self.parameters()).device

    def probability_is_imputed(self, ts):
        self.eval()
        ts_tensor = torch.tensor(ts).to(torch.float32)
        ts_tensor = ts_tensor.to(self.get_device())
        with torch.no_grad():
            scores = self.forward(ts_tensor)
        pred_is_imp = torch.sigmoid(scores)
        return pred_is_imp

    def predict_is_imputed(self, ts, threshold):
        return self.probability_is_imputed(ts) > threshold

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


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

    def fit(self, X, y):
        return self

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

    def fit(self, X, y, n_steps=10):
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
