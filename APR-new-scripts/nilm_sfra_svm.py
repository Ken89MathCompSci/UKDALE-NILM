"""
NILM SFRA + SVM classifier
==========================
Implements the mathematical method from:
Mari, S. et al. "A New NILM System Based on the SFRA Technique and Machine
Learning." Sensors 2023, 23, 5226.

Two implementations are provided:

1. `SVMFromScratch`  -- a from-scratch dual-form SVM solved via quadratic
   programming (cvxopt), directly implementing Eqs. (1)-(7) of the paper:
   the margin-maximization objective, its QP reformulation, the Lagrangian
   dual, and the polynomial kernel trick.

2. `MultiLabelApplianceSVM` -- the practical system architecture from the
   paper (Section 3.2 / Figure 8): one independent binary SVM per
   appliance, since a single multi-class SVM cannot natively output
   multiple simultaneous ON labels. Exposes the exact same interface as
   SVMFromScratch, so any of three backends can be swapped in:
     - backend="svc"     -- scikit-learn's SVC with the exact polynomial
                             kernel (Eq. 7). What the authors actually
                             used (Sec 4.1), but training cost scales
                             far worse than cubic in n on weakly-separable
                             classes -- not practical on a full dataset.
     - backend="scratch" -- SVMFromScratch, the from-scratch dual-QP
                             solver (Eqs. 1-7 exactly). Even less scalable
                             than "svc"; mainly a correctness reference.
     - backend="approx"  -- Nystroem kernel approximation (samples a
                             low-rank feature map for the same polynomial
                             kernel) + a linear SGDClassifier(loss="hinge")
                             on top. This approximates the same decision
                             boundary the paper's SVM would draw, but
                             training cost is roughly linear in n, so it
                             is the one to use when you want the model
                             trained on the full dataset rather than a
                             small random subsample -- e.g. for an
                             apples-to-apples comparison against a neural
                             baseline trained on all the data.

3. Metric functions implementing Eqs. (8)-(16): precision, recall, F1,
   and their micro/macro-averaged multi-label variants.
"""

from __future__ import annotations
import numpy as np
import cvxopt
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from dataclasses import dataclass, field

cvxopt.solvers.options["show_progress"] = False


# ---------------------------------------------------------------------------
# 1. Kernel functions (Eq. 7: polynomial kernel; linear kernel as the
#    phi(x) = x special case mentioned just before Eq. 7)
# ---------------------------------------------------------------------------

def linear_kernel(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """k(x_i, x_j) = x_i^T x_j"""
    return float(np.dot(x_i, x_j))


def polynomial_kernel(x_i: np.ndarray, x_j: np.ndarray, a: float = 1.0, b: int = 3) -> float:
    """
    Eq. (7): k(x_i, x_j) = (a + x_i^T x_j)^b

    a : additive constant (paper's "a")
    b : polynomial degree (paper's "b")
    """
    return float((a + np.dot(x_i, x_j)) ** b)


def gram_matrix(X: np.ndarray, kernel) -> np.ndarray:
    """K[i, j] = k(x_i, x_j) for all pairs, used to build the Lagrangian (Eq. 6)."""
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j])
    return K


# ---------------------------------------------------------------------------
# 2. From-scratch dual SVM (Eqs. 1-7), solved as a QP with cvxopt
#
#    Dual objective (Eq. 6), maximized over a_i >= 0, subject to
#    sum_i a_i y_i = 0. cvxopt solves QPs as *minimization* problems of
#    the form:
#        minimize    (1/2) a^T P a + q^T a
#        subject to  G a <= h
#                    A a  = c
#
#    We map the maximization in Eq. 6 to this minimization form by
#    negating, with:
#        P[i,j] = y_i y_j k(x_i, x_j)
#        q      = -1 (vector of -1s)          <-  the "sum_i a_i" term
#        G      = -I, h = 0                    <-  enforces a_i >= 0
#        A      = y^T, c = 0                   <-  enforces sum a_i y_i = 0
#    (Soft-margin slack variables add an upper bound a_i <= C, mentioned
#     in the paper's discussion of overlapping classes just before Eq. 6.)
# ---------------------------------------------------------------------------

class SVMFromScratch:
    """
    Binary SVM (+1 / -1 labels) solved in the dual form, directly
    implementing Eqs. (1)-(7) of the paper.

    Parameters
    ----------
    kernel : callable(x_i, x_j) -> float
        e.g. polynomial_kernel or linear_kernel
    C : float
        Soft-margin regularization (slack variable bound, see paper's
        discussion after Eq. 5: "data belonging to different classes
        overlap... relax some constraints by introducing slack variables").
        C = None -> hard margin (no slack).
    """

    def __init__(self, kernel=polynomial_kernel, C: float | None = 1.0):
        self.kernel = kernel
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)  # must be +1 / -1
        n_samples, _ = X.shape

        K = gram_matrix(X, self.kernel)

        # --- Build QP matrices for Eq. (6) ---
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            # Hard margin: only a_i >= 0
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # Soft margin: 0 <= a_i <= C
            G_std = -np.eye(n_samples)
            G_slack = np.eye(n_samples)
            G = cvxopt.matrix(np.vstack((G_std, G_slack)))
            h_std = np.zeros(n_samples)
            h_slack = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((h_std, h_slack)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution["x"])  # Lagrange multipliers a_i (Eq. 6)

        # --- Support vectors: samples with a_i > 0 (paper Sec 3.1: "the
        #     only samples used in the construction of the model are
        #     called support vectors") ---
        sv_tol = 1e-5
        sv_mask = a > sv_tol
        self.a_sv = a[sv_mask]
        self.X_sv = X[sv_mask]
        self.y_sv = y[sv_mask]

        # --- Bias b: average over support vectors on the margin
        #     (0 < a_i < C), using f(x_sv) with b=0 then solving for b ---
        margin_mask = sv_mask & (a < (self.C - sv_tol if self.C else np.inf))
        if not np.any(margin_mask):
            margin_mask = sv_mask  # fallback
        b_vals = []
        for i in np.where(margin_mask)[0]:
            s = sum(
                aj * yj * self.kernel(X[i], xj)
                for aj, yj, xj in zip(self.a_sv, self.y_sv, self.X_sv)
            )
            b_vals.append(y[i] - s)
        self.b = float(np.mean(b_vals))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """f(x) = w^T phi(x) + b, computed via the kernel trick (Eq. 1)."""
        X = np.asarray(X, dtype=float)
        preds = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            s = sum(
                a_i * y_i * self.kernel(x, x_i)
                for a_i, y_i, x_i in zip(self.a_sv, self.y_sv, self.X_sv)
            )
            preds[idx] = s + self.b
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """sign(f(x)) -> {-1, +1} (paper Sec 3.1, just below Eq. 1)."""
        return np.sign(self.decision_function(X))


# ---------------------------------------------------------------------------
# 3. Multi-label appliance system (paper Sec 3.2, Figure 8):
#    one binary SVM per appliance, since SVMs can't natively separate
#    3+ classes equidistantly (paper's explanation just above "To solve
#    this classification problem...").
# ---------------------------------------------------------------------------

class MultiLabelApplianceSVM:
    """
    One binary SVM per appliance. Input X is the SFRA trace (each
    frequency bin = one feature, as described in Sec 3.2: "each point
    of the trace represents a feature of the SVM").

    By default uses scikit-learn's SVC with a polynomial kernel, matching
    the paper's actual implementation (Sec 4.1: "implemented... using
    the open-source Python 3.7 from Anaconda; the machine-learning
    algorithm was developed using the Scikit-learn library").

    Parameters
    ----------
    backend : "svc" | "scratch" | "approx"
        "svc"     -- exact SVC(kernel="poly", ...), the paper's own choice.
                     Training cost scales far worse than cubic in n on
                     weakly-separable classes -- only practical on a small
                     subsample.
        "scratch" -- SVMFromScratch, the from-scratch dual-QP solver.
                     Even less scalable; correctness reference only.
        "approx"  -- Nystroem(kernel="poly", ...) + SGDClassifier(loss=
                     "hinge") pipeline. Approximates the same polynomial
                     decision boundary at roughly linear training cost, so
                     it's the one to use to train on the full dataset.
    n_components : Nystroem feature-map dimensionality ("approx" backend only).
    alpha : SGDClassifier L2 regularization strength ("approx" backend only).
    class_weight : passed straight through to SGDClassifier ("approx" backend
        only). This is a real precision/recall trade-off on rare appliance
        classes, not a free fix -- measured on both houses:
          - None (default): matches sklearn's own default. Can fully collapse
            to predicting the majority class on a severely imbalanced label
            (observed: House 2 microwave, ~1.4-1.8% ON, went to F1=0.000,
            recall=0.000 -- the model never once predicted ON).
          - "balanced": eliminates that total collapse (recall -> ~1.0 on the
            rare classes) but overcorrects precision hard in exchange
            (dishwasher/microwave/washing_machine precision dropped to
            0.03-0.21 on both houses). Net effect on macro-F1 was slightly
            *worse* on both houses (House 1 test 0.545->0.522, House 2 test
            0.493->0.478) -- it just fails in the opposite direction rather
            than being a strict improvement.
        There is no single right default here; pick based on whether a
        silently-dead detector or a trigger-happy one is the worse failure
        mode for your use case.
    use_from_scratch : deprecated alias for backend="scratch" (kept for
        backward compatibility -- overrides `backend` when True).
    """

    def __init__(
        self,
        appliance_names: list[str],
        degree: int = 3,
        coef0: float = 1.0,
        C: float = 1.0,
        use_from_scratch: bool = False,
        backend: str = "svc",
        n_components: int = 300,
        alpha: float = 1e-4,
        class_weight: str | None = None,
        random_state: int = 0,
    ):
        self.appliance_names = appliance_names
        self.backend = "scratch" if use_from_scratch else backend
        self.use_from_scratch = self.backend == "scratch"  # kept for old call sites
        self.models: dict[str, object] = {}
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.n_components = n_components
        self.alpha = alpha
        self.class_weight = class_weight
        self.random_state = random_state

    def _make_model(self):
        if self.backend == "scratch":
            kernel = lambda xi, xj: polynomial_kernel(xi, xj, a=self.coef0, b=self.degree)
            return SVMFromScratch(kernel=kernel, C=self.C)
        if self.backend == "approx":
            # Approximate the same polynomial kernel (Eq. 7) via a random
            # low-rank feature map, then fit a linear hinge-loss classifier
            # on the mapped features -- O(n) instead of the O(n^2)-O(n^3)+
            # cost of the exact kernel SVC. See class_weight docstring above
            # for the precision/recall trade-off it controls on rare classes.
            return make_pipeline(
                Nystroem(kernel="poly", degree=self.degree, coef0=self.coef0, gamma=1.0,
                         n_components=self.n_components, random_state=self.random_state),
                SGDClassifier(loss="hinge", alpha=self.alpha, max_iter=2000, tol=1e-3,
                              class_weight=self.class_weight, random_state=self.random_state),
            )
        # "svc": sklearn SVC with polynomial kernel == Eq. (7):
        # k(x_i,x_j) = (coef0 + gamma*x_i.x_j)^degree; gamma=1 recovers
        # the paper's (a + x_i^T x_j)^b exactly.
        return SVC(kernel="poly", degree=self.degree, coef0=self.coef0, gamma=1.0, C=self.C)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X : (n_samples, n_features)  SFRA traces
        Y : (n_samples, n_appliances) binary ON/OFF matrix, one column per
            appliance, values in {0,1} (converted internally to {-1,+1}
            for the from-scratch SVM; sklearn's SVC/SGDClassifier accept
            {0,1} directly).
        """
        for j, name in enumerate(self.appliance_names):
            model = self._make_model()
            y_col = Y[:, j]
            if self.backend == "scratch":
                y_col = np.where(y_col == 1, 1.0, -1.0)
            model.fit(X, y_col)
            self.models[name] = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, n_appliances) binary ON/OFF predictions."""
        preds = np.zeros((X.shape[0], len(self.appliance_names)), dtype=int)
        for j, name in enumerate(self.appliance_names):
            model = self.models[name]
            p = model.predict(X)
            if self.backend == "scratch":
                p = (p > 0).astype(int)
            preds[:, j] = p
        return preds


# ---------------------------------------------------------------------------
# 4. Metrics: Eqs. (8)-(16)
# ---------------------------------------------------------------------------

@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionCounts:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def precision_score(c: ConfusionCounts) -> float:
    """Eq. (8): Precision = TP / (TP + FP)"""
    denom = c.tp + c.fp
    return c.tp / denom if denom > 0 else 0.0


def recall_score(c: ConfusionCounts) -> float:
    """Eq. (9): Recall = TP / (TP + FN)"""
    denom = c.tp + c.fn
    return c.tp / denom if denom > 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    """Eq. (10): F1 = 2 * P * R / (P + R)"""
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Window-level MAE: mean(|y_true - y_pred|) on the binary {0,1} ON/OFF
    labels. NOT power-in-Watts MAE like the neural-network scripts in this
    repo (calculate_nilm_metrics in Source Code/utils.py) -- this SVM only
    ever outputs a window-level ON/OFF classification, never a continuous
    power estimate, so there's no Watt-scale quantity to compare. On binary
    labels this reduces exactly to the misclassified-window rate,
    (FP + FN) / N.
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    return float(np.mean(np.abs(y_true - y_pred)))


# Sub-window size for SAE, in prediction-windows (each already a
# WIN_MINUTES-long FFT window from nilm_sfra_svm_house{1,2}.py, not a raw
# 6s sample). The repo's canonical SAE (calculate_nilm_metrics in
# Source Code/utils.py) uses N=100 raw 6s samples (~10 minutes); at this
# SVM's granularity -- one row already covers a whole prediction window --
# the equivalent real-time span is reached far sooner, so the default here
# is deliberately much smaller. 10 prediction-windows x 6 min/window = 1
# hour of wall-clock time per SAE sub-window.
SAE_SUB_WINDOW = 10


def _sae_components(y_true: np.ndarray, y_pred: np.ndarray, window: int = SAE_SUB_WINDOW):
    """
    Shared implementation for sae_score and micro_average's pooled SAE.

    Matches calculate_nilm_metrics' SAE formula exactly in structure: split
    into sub-windows of `window` samples, sum the *local* |true - pred|
    aggregate difference per sub-window, THEN normalize -- this is not the
    same as a single global |sum(true) - sum(pred)| (that degenerate case
    is only correct when window >= len(y_true), i.e. num_period <= 1, and
    undercounts local mismatches that cancel out globally).

    Returns (diff_sum, denom) so callers can either divide directly
    (per-appliance) or sum diff_sum/denom across appliances first for an
    exact micro-pooled value.
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    n = len(y_true)
    num_period = int(n / window) if window > 0 else 0

    if num_period == 0:
        # Fewer samples than one sub-window -- fall back to a single
        # whole-array window rather than dividing by zero.
        diff_sum = abs(np.sum(y_true) - np.sum(y_pred))
        denom = float(n)
        return float(diff_sum), denom

    diff_sum = 0.0
    for i in range(num_period):
        s, e = i * window, (i + 1) * window
        diff_sum += abs(np.sum(y_true[s:e]) - np.sum(y_pred[s:e]))
    denom = float(window * num_period)
    return float(diff_sum), denom


def sae_score(y_true: np.ndarray, y_pred: np.ndarray, window: int = SAE_SUB_WINDOW) -> float:
    """
    Window-level SAE on the binary {0,1} ON/OFF labels, using the same
    sub-windowed-local-difference structure as calculate_nilm_metrics in
    Source Code/utils.py (see SAE_SUB_WINDOW for why the sub-window size
    differs). NOT power-in-Watts SAE like the neural-network scripts in
    this repo -- this SVM only ever outputs a window-level ON/OFF
    classification, never a continuous power estimate (same caveat as
    mae_score above).
    """
    diff_sum, denom = _sae_components(y_true, y_pred, window)
    return diff_sum / denom if denom > 0 else 0.0


def per_appliance_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, names: list[str]) -> dict:
    """Returns {appliance: {precision, recall, f1, mae, sae, tp, fp, fn, tn}}"""
    results = {}
    for j, name in enumerate(names):
        c = confusion_counts(Y_true[:, j], Y_pred[:, j])
        p = precision_score(c)
        r = recall_score(c)
        sae_diff_sum, sae_denom = _sae_components(Y_true[:, j], Y_pred[:, j])
        results[name] = {
            "precision": p,
            "recall": r,
            "f1": f1_score(p, r),
            "mae": mae_score(Y_true[:, j], Y_pred[:, j]),
            "sae": sae_diff_sum / sae_denom if sae_denom > 0 else 0.0,
            "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn,
            # Internal, used by micro_average() for exact pooling -- not a
            # metric to report on its own.
            "_sae_diff_sum": sae_diff_sum, "_sae_denom": sae_denom,
        }
    return results


def micro_average(per_app: dict) -> dict:
    """
    Eqs. (11)-(13): sum TP/FP/FN across all labels first, then compute.

    mae is pooled the same way: mae_score is linear in (FP, FN, total) on
    binary labels -- mae = (FP + FN) / N -- so summing counts across
    appliances first and applying that formula is the exact micro-pooled
    value, not an approximation.

    sae is NOT similarly reducible from TP/FP/FN alone once it's the
    sub-windowed formula in _sae_components (order within each appliance's
    sequence matters, not just the totals) -- so it's pooled by summing the
    raw per-appliance (diff_sum, denom) components from per_appliance_metrics
    before dividing, which is still exact, just needs those components
    rather than TP/FP/FN.
    """
    tp = sum(v["tp"] for v in per_app.values())
    fp = sum(v["fp"] for v in per_app.values())
    fn = sum(v["fn"] for v in per_app.values())
    tn = sum(v["tn"] for v in per_app.values())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(p, r)
    total = tp + fp + fn + tn
    mae = (fp + fn) / total if total > 0 else 0.0
    sae_diff_sum = sum(v["_sae_diff_sum"] for v in per_app.values())
    sae_denom    = sum(v["_sae_denom"]    for v in per_app.values())
    sae = sae_diff_sum / sae_denom if sae_denom > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "mae": mae, "sae": sae}


def macro_average(per_app: dict) -> dict:
    """
    Eqs. (14)-(16): average the per-label precision/recall, then compute F1.

    mae/sae are averaged the same simple unweighted way (matches the
    avgMAE/avgSAE convention already used across this repo's neural-network
    scripts, e.g. np.mean([vm[a]['mae'] for a in APPLIANCES]) in
    combined_pinn_lnn_apr_new_house1_dataset_v3.py) -- NOT derived from
    pooled counts the way micro_average's are.
    """
    n = len(per_app)
    p = sum(v["precision"] for v in per_app.values()) / n
    r = sum(v["recall"] for v in per_app.values()) / n
    f1 = f1_score(p, r)
    mae = sum(v["mae"] for v in per_app.values()) / n
    sae = sum(v["sae"] for v in per_app.values()) / n
    return {"precision": p, "recall": r, "f1": f1, "mae": mae, "sae": sae}


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)

    # ---- Synthetic "SFRA-like" data: 3 appliances, 20 frequency-bin features ----
    n_samples = 120
    n_features = 20
    appliance_names = ["hairdryer", "microwave", "lamp"]

    X = np.random.randn(n_samples, n_features)
    # Fabricate labels correlated with certain feature combinations,
    # loosely mimicking "each appliance perturbs certain frequency bins"
    Y = np.zeros((n_samples, 3), dtype=int)
    Y[:, 0] = (X[:, 0] + X[:, 1] > 0).astype(int)          # hairdryer
    Y[:, 1] = (X[:, 2] - X[:, 3] + 0.3 > 0).astype(int)     # microwave
    Y[:, 2] = (X[:, 4] * X[:, 5] > 0).astype(int)           # lamp (non-linear -> needs poly kernel)

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    print("=== sklearn-backed multi-label SVM (matches paper's actual implementation) ===")
    system = MultiLabelApplianceSVM(appliance_names, degree=3, coef0=1.0, C=1.0)
    system.fit(X_train, Y_train)
    Y_pred = system.predict(X_test)

    per_app = per_appliance_metrics(Y_test, Y_pred, appliance_names)
    for name, m in per_app.items():
        print(f"{name:12s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    mic = micro_average(per_app)
    mac = macro_average(per_app)
    print(f"\nMicro-average: P={mic['precision']:.2f} R={mic['recall']:.2f} F1={mic['f1']:.2f}")
    print(f"Macro-average: P={mac['precision']:.2f} R={mac['recall']:.2f} F1={mac['f1']:.2f}")

    print("\n=== From-scratch dual-QP SVM (directly implements Eqs. 1-7) ===")
    system_scratch = MultiLabelApplianceSVM(
        appliance_names, degree=3, coef0=1.0, C=1.0, use_from_scratch=True
    )
    system_scratch.fit(X_train, Y_train)
    Y_pred_scratch = system_scratch.predict(X_test)

    per_app_scratch = per_appliance_metrics(Y_test, Y_pred_scratch, appliance_names)
    for name, m in per_app_scratch.items():
        print(f"{name:12s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")
