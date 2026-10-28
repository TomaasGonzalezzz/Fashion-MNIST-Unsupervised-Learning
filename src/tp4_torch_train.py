import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from tp4_torch_models import get_device, make_loader


def train_autoencoder(model, X_train, X_val, epochs=150, batch_size=256, lr=1e-3, device=None):
    device = get_device() if device is None else device
    model.to(device)
    train_loader = make_loader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = F.mse_loss(x_hat, xb, reduction="mean")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_train += len(xb)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, _ = model(xb)
                loss = F.mse_loss(x_hat, xb, reduction="mean")
                val_loss += loss.item() * len(xb)
                n_val += len(xb)

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss / n_train,
                "val_mse": val_loss / n_val,
            }
        )

    return pd.DataFrame(history)


def vae_losses(x_hat, x, mu, logvar, beta):
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    loss = recon + beta * kl
    return loss, recon, kl


def train_vae(model, X_train, X_val, beta=2.0, epochs=150, batch_size=256, lr=1e-3, device=None):
    device = get_device() if device is None else device
    model.to(device)
    train_loader = make_loader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_recon = tr_kl = 0.0
        n_train = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar, _ = model(xb)
            loss, recon, kl = vae_losses(x_hat, xb, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
            tr_recon += recon.item() * len(xb)
            tr_kl += kl.item() * len(xb)
            n_train += len(xb)

        model.eval()
        va_loss = va_recon = va_kl = 0.0
        n_val = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, mu, logvar, _ = model(xb)
                loss, recon, kl = vae_losses(x_hat, xb, mu, logvar, beta)
                va_loss += loss.item() * len(xb)
                va_recon += recon.item() * len(xb)
                va_kl += kl.item() * len(xb)
                n_val += len(xb)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss / n_train,
                "train_recon": tr_recon / n_train,
                "train_kl": tr_kl / n_train,
                "val_loss": va_loss / n_val,
                "val_recon": va_recon / n_val,
                "val_kl": va_kl / n_val,
            }
        )

    return pd.DataFrame(history)


def encode_autoencoder(model, X, batch_size=512, device=None):
    device = get_device() if device is None else device
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in make_loader(X, batch_size=batch_size, shuffle=False):
            z = model.encoder(xb.to(device))
            out.append(z.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def reconstruct_autoencoder(model, X, batch_size=512, device=None):
    device = get_device() if device is None else device
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in make_loader(X, batch_size=batch_size, shuffle=False):
            x_hat, _ = model(xb.to(device))
            out.append(x_hat.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def decode_autoencoder(model, Z, batch_size=512, device=None):
    device = get_device() if device is None else device
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for (zb,) in make_loader(Z, batch_size=batch_size, shuffle=False):
            x_hat = model.decoder(zb.to(device))
            out.append(x_hat.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def encode_vae(model, X, batch_size=512, device=None):
    device = get_device() if device is None else device
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in make_loader(X, batch_size=batch_size, shuffle=False):
            mu, _ = model.encode(xb.to(device))
            out.append(mu.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def reconstruct_vae(model, X, batch_size=512, device=None):
    device = get_device() if device is None else device
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in make_loader(X, batch_size=batch_size, shuffle=False):
            mu, _ = model.encode(xb.to(device))
            # Para comparar reconstrucciones uso mu, asi no entra ruido del muestreo.
            x_hat = model.decoder(mu)
            out.append(x_hat.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def decode_vae(model, Z, batch_size=512, device=None):
    device = get_device() if device is None else device
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for (zb,) in make_loader(Z, batch_size=batch_size, shuffle=False):
            x_hat = model.decoder(zb.to(device))
            out.append(x_hat.cpu().numpy())
    return np.vstack(out).astype(np.float32)
