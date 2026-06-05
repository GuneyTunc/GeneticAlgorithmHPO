import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


def _get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_dataloaders():
    """Download Fashion-MNIST and return train / val / test DataLoaders."""
    transform = _get_transforms()

    full_train = datasets.FashionMNIST(
        root=config.DATA_DIR, train=True, download=True, transform=transform,
    )
    test_set = datasets.FashionMNIST(
        root=config.DATA_DIR, train=False, download=True, transform=transform,
    )

    val_size = int(len(full_train) * config.VAL_RATIO)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
