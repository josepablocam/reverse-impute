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
        # return shape: (num ts, num steps, 1)
        return self.layers(batch)


class SequenceEncoder(nn.Module):
    def __init__(self, hidden_size, encoder_type="bilstm"):
        super().__init__()

        if encoder_type == "bilstm":
            self.encoder = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        else:
            raise Exception("Unknown model type: {}".format(encoder_type))

    def forward(self, batch):
        # batch shape: (num ts, num steps, 1)
        # output shape: (num ts, num steps, num_dirs * hidden_size + 1)
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(2)
        output, (hn, cn) = self.encoder(batch)
        # append to each num steps the observed value as well
        # technically already captured by hidden state, but still
        # adding
        return torch.cat((output, batch), dim=2)


class ReverseImputer(nn.Module):
    def __init__(self, enc_hidden_size, pred_hidden_size):
        super().__init__()
        self.encoder = SequenceEncoder(
            enc_hidden_size,
            encoder_type="bilstm",
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
