"""
Basic SVM baseline for NILM: classification (on/off state) + regression (power draw).

Two independent SVM heads per appliance, mirroring the classification+regression
framing used in your gated dual-head ATGLNets architecture:
  - SVC  (RBF kernel) -> appliance ON/OFF state
  - SVR  (RBF kernel) -> appliance power (Watts)

Usage:
    python svm_nilm_baseline.py

Swap `load_data()` for your real UK-DALE windowed feature extraction
(e.g. the same features used for your SFRA-SVM baseline) to get a real
comparison table in the same format as your ablation logs.
"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix,
)


# ----------------------------------------------------------------------
# 1. Data loading (placeholder — replace with real UK-DALE feature windows)
# ----------------------------------------------------------------------
def load_data(appliance: str, n_samples: int = 5000, n_features: int = 16, seed: int = 0):
    """
    Placeholder synthetic data generator.

    Replace this with your real feature extraction: e.g. windowed
    aggregate power statistics (mean, std, delta, spectral features
    a la SFRA) with per-timestep appliance state (0/1) and power (W)
    as targets, same as used for the SFRA-SVM baseline.

    Returns
    -------
    X : (n_samples, n_features) feature matrix
    y_state : (n_samples,) binary on/off labels
    y_power : (n_samples,) continuous power draw (W)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    # Fake a state boundary as a nonlinear combination of features
    logits = X[:, 0] * 1.5 - X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 3]
    y_state = (logits > np.median(logits)).astype(int)

    # Power correlates with state + a couple of noisy features
    base_power = {"dishwasher": 1200, "fridge": 90, "microwave": 1100, "washing_machine": 1800}
    p0 = base_power.get(appliance, 500)
    y_power = y_state * (p0 + 50 * X[:, 4]) + rng.normal(0, 10, size=n_samples)
    y_power = np.clip(y_power, 0, None)

    return X, y_state, y_power


# ----------------------------------------------------------------------
# 2. SVM classification (state: ON/OFF)
# ----------------------------------------------------------------------
def train_svm_classifier(X_train, y_train, tune=False):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    if tune:
        # Small grid search, mirrors the GridSearchCV(C, gamma) approach
        # seen in the low-frequency SVM NILM papers.
        param_grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
        clf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"),
                            param_grid, cv=3, scoring="f1", n_jobs=-1)
        clf.fit(X_train_s, y_train)
        clf = clf.best_estimator_
    else:
        clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
        clf.fit(X_train_s, y_train)

    return clf, scaler


# ----------------------------------------------------------------------
# 3. SVM regression (power draw, Watts)
# ----------------------------------------------------------------------
def train_svm_regressor(X_train, y_train, tune=False):
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = scaler_X.transform(X_train)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()

    if tune:
        param_grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.01, 0.1]}
        reg = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=3,
                            scoring="neg_mean_absolute_error", n_jobs=-1)
        reg.fit(X_train_s, y_train_s)
        reg = reg.best_estimator_
    else:
        reg = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.05)
        reg.fit(X_train_s, y_train_s)

    return reg, scaler_X, scaler_y


# ----------------------------------------------------------------------
# 4. Evaluation, formatted like your ablation result tables
# ----------------------------------------------------------------------
def evaluate_appliance(appliance, X_test, y_state_test, y_power_test,
                        clf, clf_scaler, reg, reg_scaler_X, reg_scaler_y):
    # --- classification metrics ---
    X_test_s = clf_scaler.transform(X_test)
    y_state_pred = clf.predict(X_test_s)

    f1 = f1_score(y_state_test, y_state_pred, zero_division=0)
    prec = precision_score(y_state_test, y_state_pred, zero_division=0)
    rec = recall_score(y_state_test, y_state_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_state_test, y_state_pred, labels=[0, 1]).ravel()

    # --- regression metrics (only evaluated on true ON windows, common NILM convention) ---
    X_test_rs = reg_scaler_X.transform(X_test)
    y_power_pred_s = reg.predict(X_test_rs)
    y_power_pred = reg_scaler_y.inverse_transform(y_power_pred_s.reshape(-1, 1)).ravel()
    y_power_pred = np.clip(y_power_pred, 0, None)

    mae = np.mean(np.abs(y_power_test - y_power_pred))

    N = 100
    num_period = int(len(y_power_test) / N)
    diff = 0
    for i in range(num_period):
        diff += abs(np.sum(y_power_test[i*N:(i+1)*N]) - np.sum(y_power_pred[i*N:(i+1)*N]))
    sae = diff / (N * num_period) if num_period > 0 else 0.0

    return {
        "appliance": appliance, "f1": f1, "prec": prec, "rec": rec,
        "mae": mae, "sae": sae, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_results_table(results):
    header = f"{'Appliance':<18}{'F1':>8}{'Prec':>8}{'Rec':>8}{'MAE':>8}{'SAE':>8}{'TP':>10}{'TN':>10}{'FP':>10}{'FN':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['appliance']:<18}{r['f1']:>8.4f}{r['prec']:>8.4f}{r['rec']:>8.4f}"
              f"{r['mae']:>8.2f}{r['sae']:>8.4f}{r['tp']:>10,}{r['tn']:>10,}{r['fp']:>10,}{r['fn']:>10,}")

    n = len(results)
    print("-" * len(header))
    print(f"{'MACRO AVG':<18}"
          f"{np.mean([r['f1'] for r in results]):>8.4f}"
          f"{np.mean([r['prec'] for r in results]):>8.4f}"
          f"{np.mean([r['rec'] for r in results]):>8.4f}"
          f"{np.mean([r['mae'] for r in results]):>8.2f}"
          f"{np.mean([r['sae'] for r in results]):>8.4f}")


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main(appliances=("dishwasher", "fridge", "microwave", "washing_machine"), tune=False):
    results = []
    for appliance in appliances:
        X, y_state, y_power = load_data(appliance)
        X_train, X_test, ys_train, ys_test, yp_train, yp_test = train_test_split(
            X, y_state, y_power, test_size=0.3, random_state=42, stratify=y_state
        )

        clf, clf_scaler = train_svm_classifier(X_train, ys_train, tune=tune)
        reg, reg_scaler_X, reg_scaler_y = train_svm_regressor(X_train, yp_train, tune=tune)

        results.append(evaluate_appliance(
            appliance, X_test, ys_test, yp_test,
            clf, clf_scaler, reg, reg_scaler_X, reg_scaler_y,
        ))

    print_results_table(results)
    return results


if __name__ == "__main__":
    main(tune=False)  # set tune=True to grid-search C/gamma/epsilon (slower)
