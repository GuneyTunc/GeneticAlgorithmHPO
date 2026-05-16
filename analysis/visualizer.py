"""Visualisation helpers for MO-GA-HPO results."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import config
from ga.chromosome import decode
from models.mlp_builder import build_mlp

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ── Pareto Front ──────────────────────────────────────────────────────────────
def plot_pareto_front(front_data: list[dict], path=None):
    """Scatter plot of F1-Score vs Parameter Count for the Pareto front."""
    f1s = [d["f1_score"] for d in front_data]
    params = [d["param_count"] for d in front_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(params, f1s, c="#2563eb", edgecolors="k", s=60, zorder=3)

    sorted_pairs = sorted(zip(params, f1s))
    ax.plot([p for p, _ in sorted_pairs], [f for _, f in sorted_pairs],
            linestyle="--", color="#94a3b8", linewidth=1, zorder=2)

    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Validation F1-Score")
    ax.set_title("Pareto Front — F1-Score vs Model Size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = path or os.path.join(RESULTS_DIR, "pareto_front.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Generation Progress ───────────────────────────────────────────────────────
def plot_generation_progress(history: dict, path=None):
    """Two-panel plot: best/avg F1 and min/avg param count over generations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    gens = history["gen"]

    ax1.plot(gens, history["best_f1"], label="Best F1", marker="o", markersize=3)
    ax1.plot(gens, history["avg_f1"], label="Avg F1", linestyle="--")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("F1-Score")
    ax1.set_title("F1-Score Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, history["min_params"], label="Min Params", marker="o", markersize=3)
    ax2.plot(gens, history["avg_params"], label="Avg Params", linestyle="--")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Parameter Count")
    ax2.set_title("Model Size Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = path or os.path.join(RESULTS_DIR, "generation_progress.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Confusion Matrices ────────────────────────────────────────────────────────
def _predict(individual, loader):
    """Run inference and return (y_true, y_pred)."""
    hparams = decode(individual)
    model = build_mlp(hparams)
    model._lr = hparams["lr"]

    # Re-train briefly so we have usable weights
    # (the GA only kept fitness values, not model weights)
    device = config.DEVICE
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(config.FITNESS_EPOCHS):
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    model.cpu()
    torch.cuda.empty_cache()
    return np.array(all_labels), np.array(all_preds)


def plot_confusion(perf_champion, eff_champion, val_loader, path=None):
    """Side-by-side confusion matrices for both champions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, ind, title in [
        (ax1, perf_champion, "Performance Champion"),
        (ax2, eff_champion, "Efficiency Champion"),
    ]:
        y_true, y_pred = _predict(ind, val_loader)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=FASHION_LABELS)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
        hp = decode(ind)
        ax.set_title(
            f"{title}\n"
            f"F1={ind.fitness.values[0]:.4f} | Params={ind.fitness.values[1]:,.0f}\n"
            f"layers={hp['num_layers']} neurons={hp['neurons']} "
            f"lr={hp['lr']:.1e} act={hp['activation']} drop={hp['dropout']:.2f}"
        )

    fig.tight_layout()
    save_path = path or os.path.join(RESULTS_DIR, "confusion_matrices.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Console Summary ───────────────────────────────────────────────────────────
def print_champion_summary(perf_champion, eff_champion):
    """Print a formatted comparison of the two champions."""
    sep = "=" * 60
    print(sep)
    print("  CHAMPION SUMMARY")
    print(sep)
    for label, ind in [("Performance", perf_champion), ("Efficiency", eff_champion)]:
        hp = decode(ind)
        print(f"\n  [{label} Champion]")
        print(f"    F1-Score      : {ind.fitness.values[0]:.4f}")
        print(f"    Param Count   : {ind.fitness.values[1]:,.0f}")
        print(f"    Layers        : {hp['num_layers']}")
        print(f"    Neurons       : {hp['neurons']}")
        print(f"    Learning Rate : {hp['lr']:.6f}")
        print(f"    Activation    : {hp['activation']}")
        print(f"    Dropout       : {hp['dropout']:.3f}")
    print()
    f1_diff = abs(perf_champion.fitness.values[0] - eff_champion.fitness.values[0])
    param_diff = abs(perf_champion.fitness.values[1] - eff_champion.fitness.values[1])
    perf_params = perf_champion.fitness.values[1]
    pct = (param_diff / perf_params * 100) if perf_params > 0 else 0
    print(f"  Trade-off: {f1_diff:.4f} F1 difference for {pct:.1f}% fewer parameters")
    print(sep)
