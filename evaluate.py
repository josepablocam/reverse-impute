from argparse import ArgumentParser
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import tqdm
import pandas as pd

import models
from training import (
    TSDataset,
    get_data,
    prune_to_same_length,
)


def summary_classification_stats(y_obs, y_pred, y_probs, stats=None):
    if stats is None:
        stats = ["precision", "recall", "f1", "auc"]
    stat_funs = {
        "precision": lambda x: sklearn.metrics.precision_score(x[0], x[1]),
        "recall": lambda x: sklearn.metrics.recall_score(x[0], x[1]),
        "f1": lambda x: sklearn.metrics.f1_score(x[0], x[1]),
        "auc": lambda x: sklearn.metrics.roc_auc_score(x[0], x[2]),
    }
    results = {}
    for stat_name in stats:
        stat_fun = stat_funs[stat_name]
        try:
            results[stat_name] = stat_fun([y_obs, y_pred, y_probs])
        except ValueError:
            results[stat_name] = np.nan
    return results


def scan_for_max_f1(
        model,
        dataset,
):
    cpu = torch.device("cpu")
    model = model.to(cpu)
    model = model.eval()

    X = dataset.X
    y = dataset.y

    if isinstance(X, np.ndarray):
        X = torch.tensor(X).to(torch.float32)
        X = X.to(cpu)

    y_probs = model.probability_is_imputed(X)
    if not isinstance(y_probs, np.ndarray):
        y_probs = y_probs.cpu().numpy()
    y_probs = y_probs.flatten()
    y = y.flatten()

    results = {}
    results["auc"] = sklearn.metrics.roc_auc_score(y, y_probs)

    best_f1 = None
    best_thresh = None
    for thresh in tqdm.tqdm(np.arange(0.0, 1.0, 0.01)):
        y_hat = y_probs > thresh
        f1 = sklearn.metrics.f1_score(y, y_hat)
        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    results["best_threshold"] = best_thresh
    y_hat_best = y_probs > best_thresh
    results.update(summary_classification_stats(y, y_hat_best, y_probs))
    return results


def upper_limit_performance(
        model,
        dataset,
):
    df = dataset.df.set_index("unique_id")
    unique_ids = dataset.unique_id
    X = dataset.X
    y = dataset.y
    y_probs = model.probability_is_imputed(X)
    if not isinstance(y_probs, np.ndarray):
        y_probs = y_probs.cpu().numpy()
    nrows = X.shape[0]
    results = []

    for i in tqdm.tqdm(range(0, nrows)):
        info = {}
        subset_df = df.loc[unique_ids[i]]
        info["impute_method"] = subset_df.method.values[0]
        y_i = y[i, :]
        y_probs_i = y_probs[i, :]
        thresholds = np.linspace(0, 1.0, 11)
        f1_scores = [
            sklearn.metrics.f1_score(y_i, y_probs_i > threshold)
            for threshold in thresholds
        ]
        max_ix = np.argmax(f1_scores)
        info["threshold"] = thresholds[max_ix]
        y_hat_i = y_probs_i > thresholds[max_ix]
        info.update(
            summary_classification_stats(
                y_i, y_hat_i, y_probs_i, stats=["f1", "precision", "recall"]))
        results.append(info)
    return pd.DataFrame(results)


def compute_ts_stats(model, dataset, threshold):
    df = dataset.df.set_index("unique_id")
    unique_ids = dataset.unique_id
    X = dataset.X
    y = dataset.y
    y_probs = model.probability_is_imputed(X)
    if not isinstance(y_probs, np.ndarray):
        y_probs = y_probs.cpu().numpy()
    y_hat = y_probs > threshold
    nrows = X.shape[0]
    results = []

    for i in tqdm.tqdm(range(nrows)):
        y_i = y[i, :]
        y_hat_i = y_hat[i, :]
        y_probs_i = y_probs[i, :]
        subset_df = df.loc[unique_ids[i]]
        info = summary_classification_stats(y_i, y_hat_i, y_probs_i)
        info["unique_id"] = unique_ids[i]
        info["impute_method"] = subset_df.method.values[0]
        info["orig_mse"] = sklearn.metrics.mean_squared_error(
            subset_df.orig.values,
            subset_df.filled.values,
        )
        results.append(info)
    return pd.DataFrame(results)


def summarize_ts_stats(df):
    stats = df.groupby("impute_method")[[
        "f1", "precision", "recall", "auc", "orig_mse"
    ]].mean().reset_index()
    cts = df.groupby("impute_method").size().to_frame(name="ct").reset_index()
    return pd.merge(stats, cts, how="left", on="impute_method")


def visualize(model, threshold, df, method=None, unique_id=None, seed=None):
    fig, axes = plt.subplots(4, 1)
    if unique_id is None:
        if method is not None:
            df = df[df["method"] == method]
        if seed is None:
            seed = np.random.randint(1000)
        np.random.seed(seed)
        unique_id = np.random.choice(df.unique_id.unique(), 1)[0]
    df = df[df["unique_id"] == unique_id]

    X = torch.tensor(df.filled.values.reshape(1, -1)).to(torch.float32)
    y_probs = model.probability_is_imputed(X)
    if not isinstance(y_probs, np.ndarray):
        y_probs = y_probs.cpu().numpy()
    y_probs = y_probs.flatten()
    y_hat = y_probs > threshold

    axes[0].plot(df.orig.values, label="Ground Truth")
    axes[1].plot(df.filled.values, label="Filled")
    axes[1].scatter(
        np.where(df["mask"].values)[0],
        df.filled[df["mask"]].values,
        marker='o',
        facecolors='none',
        edgecolors='red',
        s=10,
        label="Imputed Values",
    )
    axes[2].plot(df.filled.values, label="Filled")
    axes[2].scatter(
        np.where(y_hat)[0],
        df.filled.values[np.where(y_hat)[0]],
        marker='o',
        facecolors='none',
        edgecolors='blue',
        s=20,
        label="Predicted Imputed",
    )
    axes[3].plot(y_probs, label="Probability")
    print(summary_classification_stats(df["mask"].values, y_hat, y_probs))
    return axes, df, seed


def run_evaluation(ts_data, model, baselines):
    results = []
    valid_results = scan_for_max_f1(model, ts_data["valid"])
    threshold = valid_results["best_threshold"]
    model_results = summarize_ts_stats(
        compute_ts_stats(model, ts_data["test"], threshold), )
    model_results["approach"] = "reverse-impute"
    results.append(model_results)

    for baseline_name, baseline_model in baselines.items():
        baseline_model.fit(ts_data["valid"].X, ts_data["valid"].y)
        baseline_results = summarize_ts_stats(
            compute_ts_stats(baseline_model, ts_data["test"], 0.0), )
        baseline_results["approach"] = baseline_name
        results.append(baseline_results)

    results = [r.reset_index(drop=True) for r in results]
    df = pd.concat(results, axis=0)
    return df


def sample_ts_data(ts_data, sample_n, seed=None):
    print("Sampling ts_data")
    if seed is not None:
        np.random.seed(seed)
    sampled_ts_data = {}
    for name, data in ts_data.items():
        ixs = np.arange(0, len(data))
        chosen_ixs = np.random.choice(ixs, sample_n)
        X = data.X[chosen_ixs, :]
        y = data.y[chosen_ixs, :]
        data.X = X
        data.y = y
        data.num_obs = sample_n
        sampled_ts_data[name] = data
    return sampled_ts_data


def add_white_gaussian_noise(ts_data, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noisy_ts_data = {}
    for name, ts in ts_data.items():
        noisy_df = ts.df.copy()
        nrows = ts.df.shape[0]
        noise = np.random.normal(0.0, 1.0, size=nrows)
        noisy_df["filled"] = noisy_df["filled"] + noise
        noisy_ts_data[name] = TSDataset(noisy_df)
    return noisy_ts_data


def add_noise(ts_data, method="white-gaussian", seed=None):
    print("Adding noise with method={}".format(method))
    if method == "white-gaussian":
        return add_white_gaussian_noise(ts_data, seed=seed)
    else:
        raise ValueError("Unknown noise method: {}".format(method))


def get_args():
    parser = ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "-d", "--dataset", type=str, help="Path to dataset splits")
    parser.add_argument("-c", "--csv", type=str, help="Path to csv of dataset")
    parser.add_argument(
        "-v",
        "--valid",
        type=float,
        help="Fraction of csv for validation",
        default=0.5)
    parser.add_argument(
        "-t",
        "--test",
        type=float,
        help="Fraction of csv for test",
        default=0.5)
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="Hidden size for model loading",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of RNN hidden layers for model loading",
        default=1,
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="RNG seed to split dataset", default=42)
    parser.add_argument(
        "-m", "--model", type=str, help="Path to trained model")
    parser.add_argument(
        "-b", "--baselines", type=str, nargs="+", help="Baselines to compare")
    parser.add_argument(
        "-o", "--output", type=str, help="Output df with results")
    parser.add_argument(
        "--with_noise",
        action="store_true",
        help="Add white gaussian noise to filled-in dataset",
    )
    parser.add_argument(
        "--ts_length",
        type=int,
        help="Max (and min) length of timeseries",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample n timeseries for evaluation across each split of data",
    )
    return parser.parse_args()


def main():
    args = get_args()
    model = models.ReverseImputer(
        args.hidden_size,
        args.hidden_size,
        num_layers=args.num_layers,
    )
    model.load(args.model)

    baselines = {
        "tsoutliers": models.get_tsoutliers_baseline(),
        "tsclean": models.get_tsclean_baseline(),
        "manual": models.get_manual_baseline(),
    }
    if args.baselines is not None:
        baselines = {k: m for k, m in baselines.items() if k in args.baselines}

    if args.csv is not None:
        df = pd.read_csv(args.csv)
        if args.sample is not None:
            print("Sampling here")
            np.random.seed(args.seed)
            unique_ids = np.random.choice(df.unique_id.unique(), args.sample)
            df = df[df.unique_id.isin(unique_ids)].reset_index(drop=True)
        if args.ts_length is not None:
            df = prune_to_same_length(df, args.ts_length)
        ts_data = get_data(df, args.valid, args.test, seed=args.seed)
        if args.dataset is not None:
            with open(args.dataset, "wb") as fout:
                pickle.dump(ts_data, fout)
    else:
        with open(args.dataset, "rb") as fin:
            ts_data = pickle.load(fin)

        if args.sample is not None:
            ts_data = sample_ts_data(ts_data, args.sample, seed=args.seed)

    if args.with_noise:
        ts_data = add_noise(ts_data, method="white-gaussian", seed=args.seed)



    results_df = run_evaluation(ts_data, model, baselines)
    results_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
