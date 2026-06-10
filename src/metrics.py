import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, balanced_accuracy_score, classification_report,
    brier_score_loss, log_loss, roc_auc_score
)


# Class order: 0=Repair, 1=Return to Vendor, 2=Scrap
# Cost matrix: C[true_class, action]
COST_MATRICES = {
    "base": np.array([
        #  act=Repair  act=RTV  act=Scrap
        [0,           2,       1],    # true=Repair
        [2,           0,       2],    # true=RTV
        [5,           5,       0],    # true=Scrap
    ]),
    "mild": np.array([
        [0, 1, 1],
        [1, 0, 1],
        [3, 3, 0],
    ]),
    "severe": np.array([
        [0, 3, 1],
        [3, 0, 3],
        [7, 7, 0],
    ]),
    "extreme": np.array([
        [0, 4, 1],
        [4, 0, 4],
        [10, 10, 0],
    ]),
}

SCRAP_IDX = 2


def cost_per_1000(y_true, actions, cost_matrix, review_cost=None):
    total = 0.0
    for y, a in zip(y_true, actions):
        if a == 3:  # manual review
            total += review_cost if review_cost else 0
        else:
            total += cost_matrix[y, a]
    return 1000.0 * total / len(y_true) if len(y_true) > 0 else 0


def classification_metrics(y_true, y_pred, class_names):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            results[f"{name}_precision"] = precision_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
            results[f"{name}_recall"] = recall_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
            results[f"{name}_f1"] = f1_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
    return results


def multiclass_ece(y_true, probs, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            ece += mask.sum() / len(y_true) * abs(avg_acc - avg_conf)
    return ece


def multiclass_brier(y_true, probs, n_classes=3):
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def compute_all_metrics(y_true, probs, actions, cost_matrix, class_names, review_cost=None):
    y_pred = np.argmax(probs, axis=1) if probs is not None else actions
    m = classification_metrics(y_true, y_pred, class_names)
    m["cost_per_1000"] = cost_per_1000(y_true, actions, cost_matrix, review_cost)

    if probs is not None:
        m["ece"] = multiclass_ece(y_true, probs)
        m["brier"] = multiclass_brier(y_true, probs, n_classes=len(class_names))
        try:
            m["nll"] = log_loss(y_true, probs, labels=list(range(len(class_names))))
        except:
            m["nll"] = np.nan

    auto_mask = actions != 3
    m["manual_review_rate"] = 1.0 - auto_mask.mean()
    if auto_mask.sum() > 0:
        m["cost_auto_only"] = cost_per_1000(y_true[auto_mask], actions[auto_mask], cost_matrix)
    else:
        m["cost_auto_only"] = 0
    return m


def missed_scrap_rate(y_true, actions, scrap_idx=2):
    mask = y_true == scrap_idx
    if mask.sum() == 0:
        return 0
    return (actions[mask] != scrap_idx).mean()


def false_scrap_rate(y_true, actions, scrap_idx=2):
    mask = y_true != scrap_idx
    if mask.sum() == 0:
        return 0
    return (actions[mask] == scrap_idx).mean()
