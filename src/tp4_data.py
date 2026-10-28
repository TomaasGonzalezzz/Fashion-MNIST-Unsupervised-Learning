import numpy as np
import pandas as pd


LABEL_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def load_fashion_mnist_subset(path="data/fashion_mnist_subset.csv"):
    df = pd.read_csv(path)
    y = df["label"].to_numpy(dtype=int)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return X, y, df


def as_images(X):
    return X.reshape(-1, 28, 28)


def stratified_train_eval_split(y, eval_size=0.20, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, eval_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_eval = int(round(len(idx) * eval_size))
        eval_idx.extend(idx[:n_eval])
        train_idx.extend(idx[n_eval:])

    train_idx = np.array(train_idx, dtype=int)
    eval_idx = np.array(eval_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(eval_idx)
    return train_idx, eval_idx


def stratified_sample_indices(y, n_samples=3000, seed=42):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    expected = counts / counts.sum() * n_samples
    per_class = np.floor(expected).astype(int)
    remainder = n_samples - per_class.sum()
    if remainder > 0:
        order = np.argsort(expected - per_class)[::-1]
        per_class[order[:remainder]] += 1

    chosen = []
    for cls, n_cls in zip(classes, per_class):
        idx = np.flatnonzero(y == cls)
        chosen.extend(rng.choice(idx, size=n_cls, replace=False))

    chosen = np.array(chosen, dtype=int)
    rng.shuffle(chosen)
    return chosen
