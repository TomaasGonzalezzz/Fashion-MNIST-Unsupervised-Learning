import numpy as np
import pandas as pd

from tp4_data import LABEL_NAMES
from tp4_linear import pairwise_squared_distances


def kmeans_plus_plus_init(X, k, rng):
    n = len(X)
    centers = np.empty((k, X.shape[1]), dtype=np.float32)
    first = rng.integers(n)
    centers[0] = X[first]
    closest = pairwise_squared_distances(X, centers[:1]).ravel()

    for j in range(1, k):
        total = closest.sum()
        if total <= 0:
            idx = rng.integers(n)
        else:
            probs = closest / total
            idx = rng.choice(n, p=probs)
        centers[j] = X[idx]
        new_dist = pairwise_squared_distances(X, centers[j : j + 1]).ravel()
        closest = np.minimum(closest, new_dist)
    return centers


def kmeans(X, k, max_iter=80, n_init=3, tol=1e-4, seed=42):
    X = X.astype(np.float32)
    best = None

    for init in range(n_init):
        rng = np.random.default_rng(seed + 1009 * init)
        centers = kmeans_plus_plus_init(X, k, rng)
        last_inertia = np.inf
        history = []

        for _ in range(max_iter):
            distances = pairwise_squared_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            min_dist = distances[np.arange(len(X)), labels]
            inertia = float(min_dist.sum())
            history.append(inertia)

            new_centers = np.empty_like(centers)
            counts = np.bincount(labels, minlength=k)
            farthest = np.argsort(min_dist)[::-1]
            farthest_pos = 0
            for j in range(k):
                if counts[j] == 0:
                    # Si un cluster queda vacio, lo reinicio con un punto lejano.
                    new_centers[j] = X[farthest[farthest_pos]]
                    farthest_pos += 1
                else:
                    new_centers[j] = X[labels == j].mean(axis=0)

            rel_change = abs(last_inertia - inertia) / max(last_inertia, 1.0)
            centers = new_centers
            last_inertia = inertia
            if rel_change < tol:
                break

        distances = pairwise_squared_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        min_dist = distances[np.arange(len(X)), labels]
        inertia = float(min_dist.sum())

        result = {
            "k": k,
            "centers": centers,
            "labels": labels,
            "inertia": inertia,
            "history": history,
        }
        if best is None or result["inertia"] < best["inertia"]:
            best = result

    return best


def logsumexp(A, axis=1):
    max_A = np.max(A, axis=axis, keepdims=True)
    stable = np.exp(A - max_A)
    return (max_A + np.log(stable.sum(axis=axis, keepdims=True))).squeeze(axis)


def gmm_params_from_labels(X, labels, k, min_var=1e-4):
    n, d = X.shape
    global_mean = X.mean(axis=0)
    global_var = np.maximum(X.var(axis=0), min_var)
    means = np.empty((k, d), dtype=np.float32)
    variances = np.empty((k, d), dtype=np.float32)
    weights = np.empty(k, dtype=np.float32)

    for j in range(k):
        mask = labels == j
        if mask.sum() == 0:
            means[j] = global_mean
            variances[j] = global_var
            weights[j] = 1.0 / k
        else:
            Xj = X[mask]
            means[j] = Xj.mean(axis=0)
            variances[j] = np.maximum(Xj.var(axis=0), min_var)
            weights[j] = mask.mean()

    weights = weights / weights.sum()
    return weights, means, variances


def gmm_e_step_diag(X, weights, means, variances):
    n, d = X.shape
    log_weights = np.log(np.maximum(weights, 1e-12))
    log_det = np.log(variances).sum(axis=1)
    diff = X[:, None, :] - means[None, :, :]
    maha = ((diff * diff) / variances[None, :, :]).sum(axis=2)
    log_prob = -0.5 * (d * np.log(2.0 * np.pi) + log_det[None, :] + maha)
    log_joint = log_prob + log_weights[None, :]
    log_norm = logsumexp(log_joint, axis=1)
    log_resp = log_joint - log_norm[:, None]
    return np.exp(log_resp).astype(np.float32), float(log_norm.sum())


def gmm_m_step_diag(X, resp, min_var=1e-4):
    n = len(X)
    Nj = resp.sum(axis=0) + 1e-8
    weights = (Nj / n).astype(np.float32)
    means = ((resp.T @ X) / Nj[:, None]).astype(np.float32)
    second = (resp.T @ (X * X)) / Nj[:, None]
    variances = np.maximum(second - means * means, min_var).astype(np.float32)
    return weights, means, variances


def gmm_diag(X, k, max_iter=45, n_init=2, tol=1e-4, seed=42, min_var=1e-4):
    X = X.astype(np.float32)
    best = None

    for init in range(n_init):
        km = kmeans(X, k, max_iter=35, n_init=1, seed=seed + 2017 * init)
        # Arranco desde k-Means; con medias al azar la GMM era menos estable.
        weights, means, variances = gmm_params_from_labels(X, km["labels"], k, min_var=min_var)
        previous_ll = -np.inf
        history = []

        for _ in range(max_iter):
            resp, ll = gmm_e_step_diag(X, weights, means, variances)
            weights, means, variances = gmm_m_step_diag(X, resp, min_var=min_var)
            avg_ll = ll / len(X)
            history.append(avg_ll)
            if abs(avg_ll - previous_ll) < tol:
                break
            previous_ll = avg_ll

        resp, ll = gmm_e_step_diag(X, weights, means, variances)
        labels = np.argmax(resp, axis=1)
        result = {
            "k": k,
            "weights": weights,
            "means": means,
            "variances": variances,
            "labels": labels,
            "responsibilities": resp,
            "avg_log_likelihood": ll / len(X),
            "history": history,
        }
        if best is None or result["avg_log_likelihood"] > best["avg_log_likelihood"]:
            best = result

    return best


def silhouette_score_from_distances(distances, labels):
    labels = np.asarray(labels)
    n = len(labels)
    scores = np.zeros(n, dtype=np.float32)
    clusters = np.unique(labels)

    for cls in clusters:
        idx = np.flatnonzero(labels == cls)
        if len(idx) == 1:
            continue

        intra = distances[np.ix_(idx, idx)]
        a = intra.sum(axis=1) / (len(idx) - 1)
        b = np.full(len(idx), np.inf, dtype=np.float32)

        for other in clusters:
            if other == cls:
                continue
            other_idx = np.flatnonzero(labels == other)
            mean_dist = distances[np.ix_(idx, other_idx)].mean(axis=1)
            b = np.minimum(b, mean_dist)

        denom = np.maximum(a, b)
        scores[idx] = np.where(denom > 0, (b - a) / denom, 0.0)

    return float(scores.mean())


def cluster_class_table(labels, y):
    table = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(y, name="clase"))
    for cls in sorted(LABEL_NAMES):
        if cls not in table.columns:
            table[cls] = 0
    return table[sorted(table.columns)]


def cluster_summary(labels, y):
    table = cluster_class_table(labels, y)
    rows = []
    for cluster, counts in table.iterrows():
        total = int(counts.sum())
        dominant = int(counts.idxmax())
        rows.append(
            {
                "cluster": int(cluster),
                "n": total,
                "clase_dominante": dominant,
                "nombre": LABEL_NAMES[dominant],
                "pureza": float(counts.max() / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster")


def total_purity(labels, y):
    table = cluster_class_table(labels, y)
    return float(table.max(axis=1).sum() / table.to_numpy().sum())
