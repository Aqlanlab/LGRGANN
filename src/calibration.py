import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def platt_calibrate(model, X_val, y_val, X_test):
    probs_val = model.predict_proba(X_val)
    probs_test = model.predict_proba(X_test)
    n_classes = probs_val.shape[1]
    calibrated = np.zeros_like(probs_test)
    for k in range(n_classes):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(probs_val[:, k].reshape(-1, 1), (y_val == k).astype(int))
        calibrated[:, k] = lr.predict_proba(probs_test[:, k].reshape(-1, 1))[:, 1]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    calibrated = calibrated / row_sums
    return calibrated, None


def isotonic_calibrate(model, X_val, y_val, X_test):
    probs_val = model.predict_proba(X_val)
    probs_test = model.predict_proba(X_test)
    n_classes = probs_val.shape[1]
    calibrated = np.zeros_like(probs_test)
    for k in range(n_classes):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probs_val[:, k], (y_val == k).astype(int))
        calibrated[:, k] = ir.predict(probs_test[:, k])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    calibrated = calibrated / row_sums
    return calibrated, None


def temperature_scale(probs_val, y_val, probs_test):
    def nll(T):
        scaled = softmax(np.log(np.clip(probs_val, 1e-10, 1)) / T, axis=1)
        return -np.mean(np.log(np.clip(scaled[np.arange(len(y_val)), y_val], 1e-10, 1)))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_opt = result.x
    calibrated = softmax(np.log(np.clip(probs_test, 1e-10, 1)) / T_opt, axis=1)
    return calibrated, T_opt


def calibrate_probabilities(model, X_train, y_train, X_val, y_val, X_test, method="isotonic"):
    if method == "none":
        return model.predict_proba(X_test)
    elif method == "sigmoid":
        probs, _ = platt_calibrate(model, X_val, y_val, X_test)
        return probs
    elif method == "isotonic":
        probs, _ = isotonic_calibrate(model, X_val, y_val, X_test)
        return probs
    elif method == "temperature":
        probs_val = model.predict_proba(X_val)
        probs_test = model.predict_proba(X_test)
        probs, _ = temperature_scale(probs_val, y_val, probs_test)
        return probs
    else:
        return model.predict_proba(X_test)
