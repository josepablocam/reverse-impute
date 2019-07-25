import sklearn.metrics
import torch
import numpy as np


def evaluate_model(
        model,
        X,
        y,
):
    cpu = torch.device("cpu")
    model = model.to(cpu)
    model = model.eval()
    if isinstance(X, np.ndarray):
        X = torch.tensor(X).to(torch.float32)
        X = X.to(cpu)

    with torch.no_grad():
        scores = model(X)
    y_probs = torch.sigmoid(scores)
    y_probs = y_probs.numpy().flatten()
    y = y.flatten()
    results = {
        "auc": sklearn.metrics.roc_auc_score(y, y_probs),
    }
    return results
