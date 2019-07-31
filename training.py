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
        self.unique_id = df.groupby("unique_id").size().index.values
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

    def sample(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        unique_ids = self.unique_id.copy()
        np.random.shuffle(unique_ids)
        unique_ids = unique_ids[:n]
        sampled_df = self.df[self.df.unique_id.isin(unique_ids)]
        sampled_df = sampled_df.reset_index(drop=True)
        return TSDataset(sampled_df)


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
    cats = {"train": train_ids, "valid": val_ids, "test": test_ids}
    datasets = {}
    for cat, cat_ids in cats.items():
        df_subset = df[df.ts_id.isin(cat_ids)].reset_index(drop=True)
        dataset_subset = TSDataset(df_subset)
        datasets[cat] = dataset_subset
    return datasets


class Trainer(object):
    def __init__(self, ma_max_ct=50):
        self.ma_max_ct = ma_max_ct

    def evaluate(self, model, eval_dataset, device="cpu"):
        eval_loader = data.DataLoader(
            eval_dataset,
            batch_size=len(eval_dataset),
        )
        model.eval()
        with torch.no_grad():
            X, y = next(iter(eval_loader))
            X = X.to(torch.device(device))
            y = y.to(torch.device(device))
            loss = model.compute_loss(X, y)
        model.train()
        return loss

    def train(
            self,
            model,
            datasets,
            num_iters,
            batch_size=100,
            valid_every_n_batches=None,
            device="cpu",
            tensorboard_log=None
    ):
        model = model.to(torch.device(device))

        optimizer = optim.Adam(model.parameters())

        if tensorboard_log is None:
            tensorboard_log = "runs/exp-1"
        monitor = SummaryWriter(tensorboard_log)
        iter_ct = 0

        info = []
        ma_train_losses = []

        train_dataset = datasets["train"]
        train_loader = data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
        )

        for _ in tqdm.tqdm(range(num_iters)):
            for X, y in train_loader:
                X = X.to(torch.device(device))
                y = y.to(torch.device(device))
                optimizer.zero_grad()
                batch_loss = model.compute_loss(X, y)
                batch_loss.backward()
                optimizer.step()
                iter_ct += 1

                monitor.add_scalar("data/batch_loss", batch_loss, iter_ct)
                if len(ma_train_losses) >= self.ma_max_ct:
                    ma_train_losses.pop(0)
                ma_train_losses.append(batch_loss.item())
                monitor.add_scalar(
                    "data/ma_loss",
                    np.mean(ma_train_losses),
                    iter_ct,
                )
                if valid_every_n_batches is not None and iter_ct % valid_every_n_batches == 0:
                    valid_loss = self.evaluate(model, datasets["valid"], device)
                    monitor.add_scalar(
                        "data/valid_loss",
                        valid_loss,
                        iter_ct,
                    )
                    iter_info = {}
                    iter_info["train_ma_loss"] = np.mean(ma_train_losses)
                    iter_info["val_loss"] = valid_loss.item()
                    iter_info["iter_ct"] = iter_ct
                    info.append(iter_info)

        return model, pd.DataFrame(info)
