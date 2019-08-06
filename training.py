from argparse import ArgumentParser
import os
import pickle

import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.utils.data as data
import tqdm

import models


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
        return (torch.tensor(chosen_X).to(torch.float32),
                torch.tensor(chosen_y).to(torch.float32))

    def sample(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        unique_ids = self.unique_id.copy()
        np.random.shuffle(unique_ids)
        unique_ids = unique_ids[:n]
        sampled_df = self.df[self.df.unique_id.isin(unique_ids)]
        sampled_df = sampled_df.reset_index(drop=True)
        return TSDataset(sampled_df)


def prune_to_same_length(df, length):
    # make into matrix, takes first num_obs if more
    # removes from dataset if less than num_obs available
    df = df.sort_values(["unique_id", "time"])
    unique_ids = df.unique_id.unique()
    results = []
    for uid in tqdm.tqdm(unique_ids):
        df_subset = df[df.unique_id == uid]
        if df_subset.shape[0] > length:
            df_subset = df_subset.head(length)
        if df_subset.shape[0] == length:
            results.append(df_subset)
    pruned_df = pd.concat(results, axis=0)
    return pruned_df.reset_index(drop=True)


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

    def train(self,
              model,
              datasets,
              num_iters,
              batch_size=100,
              valid_every_n_batches=None,
              device="cpu",
              tensorboard_log=None):
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
                    valid_loss = self.evaluate(model, datasets["valid"],
                                               device)
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


def get_args():
    parser = ArgumentParser(description="Train model")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="CSV with training data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--valid",
        type=float,
        help="Fraction of data for validation",
        default=0.2,
    )
    parser.add_argument(
        "--test",
        type=float,
        help="Fraction of data for test",
        default=0.1,
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="Size of hidden layers",
        default=50,
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        help="Number of iterations over dataset",
        default=20,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Size of each mini-batch",
        default=100,
    )
    parser.add_argument(
        "--valid_every_n_batches",
        type=int,
        help="Validate model every n batches",
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed for data splitting",
    )
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.input)
    print("Preparing data split")
    ts_data = get_data(
        df,
        frac_valid=args.valid,
        frac_test=args.test,
        seed=args.seed,
    )
    model = models.ReverseImputer(
        enc_hidden_size=args.hidden_size,
        pred_hidden_size=args.hidden_size,
    )
    trainer = Trainer()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model, info_df = trainer.train(
        model,
        ts_data,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        valid_every_n_batches=args.valid_every_n_batches,
        device=device,
    )
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    model.save(os.path.join(args.output, "model.pth"))
    with open(os.path.join(args.output, "dataset.pkl"), "wb") as fout:
        pickle.dump(ts_data, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
