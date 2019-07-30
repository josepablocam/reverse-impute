import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import tqdm
import pandas as pd


def summary_classification_stats(y_obs, y_pred, y_probs):
    return {
        "precision": sklearn.metrics.precision_score(y_obs, y_pred),
        "recall": sklearn.metrics.recall_score(y_obs, y_pred),
        "f1": sklearn.metrics.f1_score(y_obs, y_pred),
        "auc_score": sklearn.metrics.roc_auc_score(y_obs, y_probs),
    }


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
    if not instance(y_probs, np.ndarray):
        y_probs = y_probs.numpy()
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


def compute_ts_stats(model, dataset, threshold):
    df = dataset.df.set_index("unique_id")
    unique_ids = dataset.unique_id
    X = dataset.X
    y = dataset.y
    y_probs = model.probability_is_imputed(X)
    if not isinstance(y_probs, np.ndarray):
        y_probs = y_probs.numpy()
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
    return df.groupby("impute_method")[[
        "f1", "precision", "recall", "auc_score", "orig_mse"
    ]].mean()


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

    X = torch.tensor(df.filled.values).to(torch.float32)
    y_probs = model.probability_is_imputed(X)
    if not isinstance(y_probs, np.ndarray):
        y_probs = y_probs.numpy()
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
