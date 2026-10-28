import numpy as np


def fit_standardizer(X, eps=1e-6):
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < eps] = 1.0
    return {"mean": mean, "std": std}


def transform_standardizer(X, standardizer):
    return ((X - standardizer["mean"]) / standardizer["std"]).astype(np.float32)


def inverse_standardizer(X_std, standardizer):
    return (X_std * standardizer["std"] + standardizer["mean"]).astype(np.float32)


def fit_pca(X_std):
    X_work = X_std.astype(np.float64)
    cov = (X_work.T @ X_work) / (len(X_work) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].astype(np.float32)

    total = eigenvalues.sum()
    explained_ratio = eigenvalues / total
    cumulative_ratio = np.cumsum(explained_ratio)

    return {
        "components": components,
        "eigenvalues": eigenvalues.astype(np.float32),
        "explained_ratio": explained_ratio.astype(np.float32),
        "cumulative_ratio": cumulative_ratio.astype(np.float32),
    }


def n_components_for_variance(pca, target=0.90):
    return int(np.searchsorted(pca["cumulative_ratio"], target) + 1)


def pca_transform(X_std, pca, n_components):
    U_m = pca["components"][:, :n_components]
    return (X_std @ U_m).astype(np.float32)


def pca_inverse_transform(Z, pca, n_components):
    U_m = pca["components"][:, :n_components]
    return (Z @ U_m.T).astype(np.float32)


def reconstruction_mse(X, X_hat):
    return float(np.mean((X - X_hat) ** 2))


def pairwise_squared_distances(A, B):
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    dist = A2 + B2 - 2.0 * (A @ B.T)
    return np.maximum(dist, 0.0)


def pairwise_distances(X, chunk_size=600):
    X = X.astype(np.float32)
    n = len(X)
    distances = np.empty((n, n), dtype=np.float32)
    # Lo hago por bloques para no duplicar memoria de mas con 3000 puntos.
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        distances[start:end] = np.sqrt(pairwise_squared_distances(X[start:end], X))
    return distances
