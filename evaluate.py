import sklearn.metrics
import torch


def evaluate_model(
        model,
        X_test,
        y_test,
):
    cpu = torch.device("cpu")
    model = model.to(cpu)
    model = model.eval()
    X_test = X_test.to(cpu)
    y_test = y_test.to(cpu)

    scores = model(X_test)
    y_probs = torch.sigmoid(scores)

    y_probs = y_probs.numpy().flatten()
    y_test = y_test.numpy().flatten()
    results = {
        "auc": metrics.roc_auc_score(y_test, y_probs),
    }
    return results
