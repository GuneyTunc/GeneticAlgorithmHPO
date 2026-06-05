"""Full success and error metric evaluation of GA champions."""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.data_loader import get_dataloaders
from models.mlp_builder import build_mlp, count_parameters

CLASS_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

CHAMPIONS = {
    "Performance": {
        "num_layers": 2,
        "neurons": 512,
        "lr": 0.000939,
        "activation": "relu",
        "dropout": 0.105,
    },
    "Efficiency": {
        "num_layers": 1,
        "neurons": 16,
        "lr": 0.002247,
        "activation": "elu",
        "dropout": 0.0,
    },
}


def _train_model(model, train_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=model._lr)
    criterion = nn.CrossEntropyLoss()
    non_block = device.type == "cuda"
    model.train()
    for _ in range(config.FITNESS_EPOCHS):
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device, non_blocking=non_block)
            labels = labels.to(device, non_blocking=non_block)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def _collect_predictions(model, loader, device):
    model.eval()
    non_block = device.type == "cuda"
    y_true, y_pred, y_proba = [], [], []
    inference_times = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device, non_blocking=non_block)
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_proba.append(probs)
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    return (
        np.array(y_true),
        np.array(y_pred),
        np.vstack(y_proba),
        (sum(inference_times) / len(inference_times)) * 1000,
    )


def _per_class_error_rates(y_true, y_pred, n_classes):
    """Per-class misclassification rate = 1 - recall for that class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    rates = {}
    for i, label in enumerate(CLASS_LABELS):
        support = cm[i].sum()
        correct = cm[i, i]
        rate = 1.0 - (correct / support) if support > 0 else 0.0
        rates[label] = {
            "error_rate": float(rate),
            "misclassified": int(support - correct),
            "support": int(support),
        }
    return rates


def _macro_fpr_fnr(y_true, y_pred, n_classes):
    """Macro-averaged false positive and false negative rates."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fprs, fnrs = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        fnrs.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
    return float(np.mean(fprs)), float(np.mean(fnrs))


def compute_metrics(y_true, y_pred, y_proba, inference_ms, param_count):
    n_classes = config.NUM_CLASSES
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    accuracy = accuracy_score(y_true, y_pred)
    macro_fpr, macro_fnr = _macro_fpr_fnr(y_true, y_pred, n_classes)
    misclassified = int((y_true != y_pred).sum())
    total = int(len(y_true))

    success = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "jaccard_macro": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc_ovr_macro": float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")),
        "param_count": int(param_count),
        "inference_ms_per_batch": float(inference_ms),
        "inference_ms_per_sample": float(inference_ms / config.BATCH_SIZE),
    }

    error = {
        "error_rate": float(1.0 - accuracy),
        "misclassification_rate": float(1.0 - accuracy),
        "top1_error_rate": float(1.0 - accuracy),
        "zero_one_loss": float(zero_one_loss(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=list(range(n_classes)))),
        "brier_score_macro": float(np.mean([
            brier_score_loss(y_bin[:, i], y_proba[:, i]) for i in range(n_classes)
        ])),
        "macro_false_positive_rate": macro_fpr,
        "macro_false_negative_rate": macro_fnr,
        "misclassified_count": misclassified,
        "total_samples": total,
        "correct_count": total - misclassified,
        "per_class_error": _per_class_error_rates(y_true, y_pred, n_classes),
    }

    return {"success": success, "error": error}


def evaluate_champion(hparams, train_loader, val_loader, test_loader, device):
    model = build_mlp(hparams)
    model._lr = hparams["lr"]
    param_count = count_parameters(model)
    model = model.to(device)

    _train_model(model, train_loader, device)

    results = {}
    for split_name, loader in [("validation", val_loader), ("test", test_loader)]:
        y_true, y_pred, y_proba, infer_ms = _collect_predictions(model, loader, device)
        ms = compute_metrics(y_true, y_pred, y_proba, infer_ms, param_count)
        ms["classification_report"] = classification_report(
            y_true, y_pred, target_names=CLASS_LABELS, zero_division=0,
        )
        results[split_name] = ms

    model.cpu()
    del model
    torch.cuda.empty_cache()
    return results


def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    device = config.DEVICE
    output = {"champions": {}, "ga_fitness_note": (
        "GA fitness F1 (validation, stored during evolution): "
        "Performance=0.8852, Efficiency=0.8562"
    )}

    for name, hparams in CHAMPIONS.items():
        print(f"Evaluating {name} champion ...")
        output["champions"][name] = {
            "hparams": hparams,
            **evaluate_champion(hparams, train_loader, val_loader, test_loader, device),
        }

    out_json = os.path.join("results", "champion_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    out_md = os.path.join("results", "champion_error_metrics.md")
    _write_report_md(output, out_md)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}\n")
    _print_summary(output)


def _write_report_md(output, path):
    lines = [
        "# Sampiyon Modeller - Basari ve Hata Metrikleri\n",
        "Fitness epoch: 5 | Optimizer: Adam | Batch: 512\n\n",
    ]
    for name, data in output["champions"].items():
        lines.append(f"## {name} Champion\n\n")
        hp = data["hparams"]
        lines.append(
            f"Mimari: {hp['num_layers']} katman, {hp['neurons']} noron, "
            f"{hp['activation']}, lr={hp['lr']:.6f}, dropout={hp['dropout']:.3f}\n\n"
        )
        for split in ("validation", "test"):
            s = data[split]["success"]
            e = data[split]["error"]
            lines.append(f"### {split.capitalize()} seti\n\n")
            lines.append("| Metrik | Deger |\n|--------|-------|\n")
            lines.append(f"| Accuracy | {s['accuracy']:.4f} ({s['accuracy']*100:.2f}%) |\n")
            lines.append(f"| F1 (macro) | {s['f1_macro']:.4f} |\n")
            lines.append(f"| Hata orani (1-accuracy) | {e['error_rate']:.4f} ({e['error_rate']*100:.2f}%) |\n")
            lines.append(f"| Hamming loss | {e['hamming_loss']:.4f} |\n")
            lines.append(f"| Log loss | {e['log_loss']:.4f} |\n")
            lines.append(f"| Zero-one loss | {e['zero_one_loss']:.4f} |\n")
            lines.append(f"| Brier score (macro) | {e['brier_score_macro']:.4f} |\n")
            lines.append(f"| Macro FPR | {e['macro_false_positive_rate']:.4f} |\n")
            lines.append(f"| Macro FNR | {e['macro_false_negative_rate']:.4f} |\n")
            lines.append(
                f"| Yanlis siniflandirma | {e['misclassified_count']}/{e['total_samples']} |\n\n"
            )
            lines.append("**Sinif bazli hata orani (1 - recall):**\n\n")
            lines.append("| Sinif | Hata orani | Yanlis | Destek |\n")
            lines.append("|-------|------------|--------|--------|\n")
            for cls, info in e["per_class_error"].items():
                lines.append(
                    f"| {cls} | {info['error_rate']*100:.1f}% | "
                    f"{info['misclassified']} | {info['support']} |\n"
                )
            lines.append("\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _print_summary(output):
    for name, data in output["champions"].items():
        t = data["test"]
        print(f"=== {name} (TEST) ===")
        print(f"  Accuracy     : {t['success']['accuracy']*100:.2f}%")
        print(f"  Error rate   : {t['error']['error_rate']*100:.2f}%")
        print(f"  F1 macro     : {t['success']['f1_macro']*100:.2f}%")
        print(f"  Log loss     : {t['error']['log_loss']:.4f}")
        print(f"  Misclassified: {t['error']['misclassified_count']}/{t['error']['total_samples']}")
        print()


if __name__ == "__main__":
    main()
