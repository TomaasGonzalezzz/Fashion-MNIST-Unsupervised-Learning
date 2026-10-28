from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tp4_data import LABEL_NAMES


def set_plot_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["axes.titleweight"] = "bold"


def save_figure(fig, filename, figures_dir="figures"):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)
    path = figures_dir / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def to_images_for_plot(X):
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 784:
        return X.reshape(-1, 28, 28)
    return X


def plot_images(X, labels=None, title=None, n_cols=5):
    images = to_images_for_plot(X)
    n = len(images)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 2.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        if labels is not None:
            label = int(labels[i])
            ax.set_title(f"{label}: {LABEL_NAMES[label]}", fontsize=8)

    if title:
        fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_class_distribution(y, title=None):
    counts = pd.Series(y).value_counts().sort_index()
    names = [LABEL_NAMES[int(i)] for i in counts.index]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, counts.values, color="#427AA1")
    ax.set_ylabel("Cantidad de imagenes")
    ax.set_xlabel("Clase")
    ax.set_title(title or "Distribucion de clases")
    ax.tick_params(axis="x", rotation=35)
    for pos, value in enumerate(counts.values):
        ax.text(pos, value + counts.max() * 0.01, str(value), ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_samples_by_class(X, y, classes, n_per_class=7, seed=42, title=None):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(1.55 * n_per_class, 1.75 * len(classes)))
    axes = np.asarray(axes)

    for row, cls in enumerate(classes):
        idx = np.flatnonzero(y == cls)
        chosen = rng.choice(idx, size=n_per_class, replace=False)
        for col, image_idx in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(X[image_idx].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"{cls}: {LABEL_NAMES[int(cls)]}", rotation=0, ha="right", va="center", fontsize=9)

    if title:
        fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    return fig


def plot_average_images_by_class(X, y, title=None):
    fig, axes = plt.subplots(2, 5, figsize=(9, 4.2))
    axes = axes.ravel()

    for cls, ax in enumerate(axes):
        mean_img = X[y == cls].mean(axis=0).reshape(28, 28)
        ax.imshow(mean_img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{cls}: {LABEL_NAMES[cls]}", fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_intensity_by_class(X, y, title=None):
    mean_intensity = X.mean(axis=1)
    df = pd.DataFrame(
        {
            "intensidad_media": mean_intensity,
            "clase": [LABEL_NAMES[int(label)] for label in y],
        }
    )
    order = [LABEL_NAMES[i] for i in range(10)]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    sns.boxplot(data=df, x="clase", y="intensidad_media", order=order, ax=ax, color="#F2CC8F")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Intensidad media por imagen")
    ax.set_title(title or "Intensidad media por clase")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def plot_pca_variance(cumulative_ratio, k_90, title=None):
    xs = np.arange(1, len(cumulative_ratio) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, cumulative_ratio, color="#2A9D8F", lw=2)
    ax.axhline(0.90, color="#E76F51", ls="--", lw=1.5, label="90%")
    ax.axvline(k_90, color="#264653", ls="--", lw=1.5, label=f"K = {k_90}")
    ax.set_xlabel("Numero de componentes principales")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or "Varianza explicada acumulada")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reconstruction_rows(original, reconstructions, row_names, title=None, n=10):
    rows = [original[:n]] + [reconstructions[name][:n] for name in row_names]
    labels = ["Original"] + row_names
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(1.35 * n, 1.45 * n_rows))
    axes = np.asarray(axes)

    for row_idx, row_images in enumerate(rows):
        for col in range(n):
            ax = axes[row_idx, col]
            ax.imshow(row_images[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(labels[row_idx], rotation=0, ha="right", va="center", fontsize=9)

    if title:
        fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    return fig


def plot_ae_history(history, title=None):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(
        history["epoch"],
        history["train_mse"],
        marker="o",
        markersize=2,
        markeredgewidth=0,
        linewidth=1.5,
        label="Train",
    )
    ax.plot(
        history["epoch"],
        history["val_mse"],
        marker="o",
        markersize=2,
        markeredgewidth=0,
        linewidth=1.5,
        label="Validacion",
    )
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE")
    ax.set_title(title or "Entrenamiento AE")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_vae_history(history, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        ("loss", "Loss total"),
        ("recon", "Reconstruccion BCE"),
        ("kl", "KL"),
    ]

    for ax, (suffix, label) in zip(axes, panels):
        ax.plot(
            history["epoch"],
            history[f"train_{suffix}"],
            marker="o",
            markersize=2,
            markeredgewidth=0,
            linewidth=1.4,
            label="Train",
        )
        ax.plot(
            history["epoch"],
            history[f"val_{suffix}"],
            marker="o",
            markersize=2,
            markeredgewidth=0,
            linewidth=1.4,
            label="Validacion",
        )
        ax.set_xlabel("Epoca")
        ax.set_title(label)
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, y=1.05, fontsize=13)
    fig.tight_layout()
    return fig


def plot_cluster_metrics(results, metric_col, gain_col, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(results["k"], results[metric_col], marker="o", color="#3D5A80")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel(metric_col)
    axes[0].set_title("Metrica principal")

    axes[1].plot(results["k"], results[gain_col], marker="o", color="#EE6C4D", label="Ganancia")
    axes[1].plot(results["k"], results["silhouette"], marker="s", color="#2A9D8F", label="Silhouette")
    axes[1].set_xlabel("K")
    axes[1].set_title("Ganancia marginal y Silhouette")
    axes[1].legend()

    if title:
        fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    return fig


def plot_tsne(Y, color_values, title=None, palette="tab10", colorbar_label=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=color_values, s=12, cmap=palette, alpha=0.82, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or "t-SNE")
    if colorbar_label:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig


def plot_cluster_heatmap(table, title=None):
    row_sums = table.sum(axis=1).replace(0, 1)
    normalized = table.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(normalized.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xlabel("Clase real")
    ax.set_ylabel("Cluster")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([LABEL_NAMES[int(c)] for c in table.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_title(title or "Composicion de clusters")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporcion dentro del cluster")
    fig.tight_layout()
    return fig


def plot_cluster_sizes(summary, title=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["cluster"].astype(str), summary["n"], color="#81B29A")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cantidad de muestras")
    ax.set_title(title or "Tamanio de clusters")
    fig.tight_layout()
    return fig
