import numpy as np
from scipy.stats import entropy as sp_entropy


def bayes_risk_decision(probs, cost_matrix):
    risk = probs @ cost_matrix
    actions = np.argmin(risk, axis=1)
    return actions, risk


def argmax_decision(probs):
    return np.argmax(probs, axis=1)


def defer_decision(probs, cost_matrix, review_cost, policy="risk",
                   threshold=None, capacity=None):
    actions, risk = bayes_risk_decision(probs, cost_matrix)
    best_risk = np.min(risk, axis=1)
    n = len(probs)

    if policy == "risk":
        defer = best_risk > review_cost
    elif policy == "confidence":
        defer = np.max(probs, axis=1) < (threshold or 0.6)
    elif policy == "entropy":
        ent = sp_entropy(probs.T)
        defer = ent > (threshold or 0.8)
    elif policy == "risk_margin":
        sorted_risk = np.sort(risk, axis=1)
        margin = sorted_risk[:, 1] - sorted_risk[:, 0]
        defer = margin < (threshold or 0.5)
    else:
        defer = np.zeros(n, dtype=bool)

    if capacity is not None and capacity < 1.0:
        max_defer = int(capacity * n)
        if defer.sum() > max_defer:
            scores = best_risk.copy()
            scores[~defer] = -np.inf
            top_k = np.argsort(scores)[-max_defer:]
            new_defer = np.zeros(n, dtype=bool)
            new_defer[top_k] = True
            defer = new_defer

    final_actions = actions.copy()
    final_actions[defer] = 3  # manual review
    return final_actions, risk, defer
