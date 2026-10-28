import numpy as np
import pandas as pd

from tp4_linear import pairwise_squared_distances


def binary_search_perplexity(D_i, target_log_perplexity, max_iter=35, tol=1e-4):
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf

    for _ in range(max_iter):
        P_i = np.exp(-D_i * beta)
        sum_P = np.maximum(P_i.sum(), 1e-12)
        H = np.log(sum_P) + beta * np.sum(D_i * P_i) / sum_P
        H_diff = H - target_log_perplexity

        if abs(H_diff) < tol:
            break
        if H_diff > 0:
            beta_min = beta
            beta = beta * 2.0 if np.isinf(beta_max) else (beta + beta_max) / 2.0
        else:
            beta_max = beta
            beta = beta / 2.0 if np.isinf(beta_min) else (beta + beta_min) / 2.0

    P_i = np.exp(-D_i * beta)
    P_i = P_i / np.maximum(P_i.sum(), 1e-12)
    return P_i


def tsne_embedding(
    X,
    perplexity=30.0,
    n_iter=350,
    learning_rate=200.0,
    early_exaggeration=12.0,
    exaggeration_iters=80,
    seed=42,
):
    X = X.astype(np.float32)
    n = len(X)
    rng = np.random.default_rng(seed)

    # Para 3000 puntos, la version exacta todavia entra en memoria.
    distances = pairwise_squared_distances(X, X).astype(np.float32)
    np.fill_diagonal(distances, np.inf)

    P = np.zeros((n, n), dtype=np.float32)
    target = np.log(perplexity)
    for i in range(n):
        mask = np.isfinite(distances[i])
        P[i, mask] = binary_search_perplexity(distances[i, mask], target)

    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12).astype(np.float32)

    Y = rng.normal(0.0, 1e-4, size=(n, 2)).astype(np.float32)
    velocity = np.zeros_like(Y)
    history = []

    for it in range(n_iter):
        sum_Y = np.sum(Y * Y, axis=1, keepdims=True)
        dist_Y = np.maximum(sum_Y + sum_Y.T - 2.0 * (Y @ Y.T), 0.0)
        num = 1.0 / (1.0 + dist_Y)
        np.fill_diagonal(num, 0.0)
        Q = num / np.maximum(num.sum(), 1e-12)
        Q = np.maximum(Q, 1e-12).astype(np.float32)

        P_use = P * early_exaggeration if it < exaggeration_iters else P
        W = (P_use - Q) * num
        row_sum = W.sum(axis=1, keepdims=True)
        grad = 4.0 * (Y * row_sum - W @ Y)

        momentum = 0.5 if it < 250 else 0.8
        velocity = momentum * velocity - learning_rate * grad
        Y = Y + velocity
        Y = Y - Y.mean(axis=0, keepdims=True)

        if (it + 1) % 50 == 0 or it == 0:
            kl = np.sum(P * np.log(P / Q))
            history.append({"iter": it + 1, "kl": float(kl)})

    return Y.astype(np.float32), pd.DataFrame(history)
