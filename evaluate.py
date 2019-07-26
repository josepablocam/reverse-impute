import numpy as np
import sklearn.metrics
import torch
import tqdm


def summary_classification_stats(y_obs, y_pred):
    return {
        "precision": sklearn.metrics.precision_score(y_obs, y_pred),
        "recall": sklearn.metrics.precision_score(y_obs, y_pred),
        "f1": sklearn.metrics.f1_score(y_obs, y_pred),
    }


def evaluate_model(
        model,
        dataset,
        max_f1=True,
):
    cpu = torch.device("cpu")
    model = model.to(cpu)
    model = model.eval()

    X = dataset.X
    y = dataset.y

    if isinstance(X, np.ndarray):
        X = torch.tensor(X).to(torch.float32)
        X = X.to(cpu)

    with torch.no_grad():
        scores = model(X)
    y_probs = torch.sigmoid(scores)
    y_probs = y_probs.numpy().flatten()
    y = y.flatten()

    results = {}
    results["auc"] = sklearn.metrics.roc_auc_score(y, y_probs)

    if max_f1:
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
        results.update(summary_classification_stats(y, y_hat_best))
    return results


def compute_ts_stats(repairer, dataset, threshold):
    df = dataset.df.set_index("unique_id")
    unique_ids = dataset.unique_id
    X = dataset.X
    y = dataset.y
    yhat = repairer.predict_is_imputed(X, threshold=threshold)
    nrows = X.shape[0]
    results = []

    for i in tqdm.tqdm(range(nrows)):
        y_obs = y[i, :]
        y_pred = y_hat[i, :]
        info = summary_classification_stats(y_obs, y_pred)
        info["unique_id"] = unique_ids[i]
        info["impute_method"] = df.loc[unique_ids[i]].method
        results.append(info)
    return pd.DataFrame(results)


def summarize_ts_stats(df):
    return df.groupby("impute_method")[["f1", "precision", "recall"]].mean()


# def compute_per_ts_stats(
#         repairer,
#         df,
#         consider_methods,
#         use_method,
#         threshold,
# ):
#     results = []
#     df = df.method.isin(consider_methods)
#     df_ix = df.set_index("unique_id")
#     for unique_id in tqdm.tqdm(df.unique_id.unique()):
#         stats = {}
#         filled = df.loc[unique_id].filled.values
#         no_missing = df.loc[unique_id].orig.values
#         with_missing = df.loc[unique_id].with_missing.values
#
#         pred_is_imputed = repairer.predict_is_imputed(filled, threshold)
#         gold_is_imputed = np.isnan(with_missing)
#
#         prec_score = sklearn.metrics.precision_score(
#             gold_is_imputed,
#             pred_is_imputed,
#         )
#         recall_score = sklearn.metrics.recall_score(
#             gold_is_imputed, pred_is_imputed
#         )
#
#         repaired = repairer.reimpute(filled, use_method, threshold)
#         mse_before = sklearn.metrics.mean_squared_error(no_missing, filled)
#         mse_after = sklearn.metrics.mean_squared_error(no_missing, repaired)
#         stats["unique_id"] = unique_id
#         stats["method"] = df.loc[unique_id].method
#         stats["precision"] = prec_score
#         stats["recall"] = recall_score
#         stats["mse_before"] = mse_before
#         stats["mse_after"] = mse_after
#         stats["mse_change"] = mse_after - mse_before
#         results.append(stats)
#     return pd.DataFrame(results)
