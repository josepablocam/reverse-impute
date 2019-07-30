import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import tqdm


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

    def probability_is_imputed(self, ts):
        self.eval()
        ts_tensor = torch.tensor(ts).to(torch.float32)
        with torch.no_grad():
            scores = self.forward(ts_tensor)
        pred_is_imp = torch.sigmoid(scores)
        return pred_is_imp

    def predict_is_imputed(self, ts, threshold):
        return self.probability_is_imputed(ts) > threshold
