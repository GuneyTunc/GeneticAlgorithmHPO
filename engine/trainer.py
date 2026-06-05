import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import config


def train_and_evaluate(model: nn.Module, train_loader, val_loader):
    """Train *model* for ``config.FITNESS_EPOCHS`` epochs and return metrics.

    Returns
    -------
    f1 : float
        Macro-averaged F1 score on the validation set.
    param_count : int
        Total trainable parameter count.
    inference_ms : float
        Average per-batch inference time on the validation set (milliseconds).
    """
    device = config.DEVICE
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model._lr)
    criterion = nn.CrossEntropyLoss()

    # ── Training ──────────────────────────────────────────────────────────
    non_block = device.type == "cuda"
    model.train()
    for _ in range(config.FITNESS_EPOCHS):
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device, non_blocking=non_block)
            labels = labels.to(device, non_blocking=non_block)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # ── Validation ────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    inference_times: list[float] = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(images.size(0), -1).to(device, non_blocking=non_block)

            start = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    inference_ms = (sum(inference_times) / len(inference_times)) * 1000

    # Free GPU memory
    model.cpu()
    del model
    torch.cuda.empty_cache()

    return f1, param_count, inference_ms
