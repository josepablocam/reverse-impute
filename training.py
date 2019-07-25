import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.utils.data as data
import tqdm


def get_matrix_vals(df, key_col, val_col):
    lists = df.groupby(key_col)[val_col].apply(list).values
    return np.array([np.array(l) for l in lists])


class TSDataset(data.Dataset):
    def __init__(self, df):
        df = df.sort_values(["unique_id", "time"], ascending=True)
        self.df = df
        self.X = get_matrix_vals(df, "unique_id", "filled")
        self.y = get_matrix_vals(df, "unique_id", "mask")
        self.num_obs = self.X.shape[0]

    def __len__(self):
        return self.num_obs

    def __getitem__(self, ix):
        chosen_X = self.X[ix]
        chosen_y = self.y[ix]
        return (
            torch.tensor(chosen_X).to(torch.float32),
            torch.tensor(chosen_y).to(torch.float32)
        )


def get_data(df, frac_valid, frac_test, seed=None):
    if isinstance(df, str):
        df = pd.read_csv(df)
    # need to split data along ts_id
    # otherwise can memorize
    ts_ids = df.ts_id.unique()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(ts_ids)
    n = len(ts_ids)
    num_valid = int(frac_valid * n)
    num_train = n - num_valid - int(frac_test * n)

    train_ids = ts_ids[:num_train]
    val_ids = ts_ids[num_train:(num_train + num_valid)]
    test_ids = ts_ids[(num_train + num_valid):]
    cats = {"train": train_ids, "val": val_ids, "test": test_ids}
    datasets = {}
    for cat, cat_ids in cats.items():
        df_subset = df[df.ts_id.isin(cat_ids)].reset_index(drop=True)
        dataset_subset = TSDataset(df_subset)
        datasets[cat] = dataset_subset
    return datasets


class Trainer(object):
    def __init__(self, num_iters, batch_size):
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.ma_max_ct = 100

    def evaluate(self, model, eval_dataset):
        eval_loader = data.DataLoader(
            eval_dataset,
            batch_size=len(eval_dataset),
        )
        model.eval()
        with torch.no_gradient():
            X, y = next(iter(eval_loader))
            loss = model.compute_loss(X, y)
        model.train()
        return loss

    def train(
            self,
            model,
            datasets,
            valid_every_n_batches=None,
    ):
        optimizer = optim.Adam(model.parameters())

        monitor = SummaryWriter()
        iter_ct = 0
        ma_losses = []

        train_dataset = datasets["train"]
        train_loader = data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
        )

        for _ in tqdm.tqdm(range(self.num_iters)):
            for X, y in train_loader:
                optimizer.zero_grad()
                batch_loss = model.compute_loss(X, y)
                batch_loss.backward()
                optimizer.step()
                iter_ct += 1
                monitor.add_scalar("data/batch_loss", batch_loss, iter_ct)
                if len(ma_losses) >= self.ma_max_ct:
                    ma_losses = ma_losses.pop(0)
                ma_losses.append(batch_loss)
                monitor.add_scalar(
                    "data/ma_loss",
                    torch.mean(ma_losses),
                    iter_ct,
                )
                if valid_every_n_batches is not None and iter_ct % valid_every_n_batches == 0:
                    valid_loss = self.evaluate(model, datasets["valid"])
                    monitor.add_scalar(
                        "data/valid_loss",
                        valid_loss,
                        iter_ct,
                    )
        return model
