import torch.nn as nn

import config


_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def build_mlp(hparams: dict) -> nn.Sequential:
    """Build a fully-connected network from a decoded hyperparameter dict.

    Parameters
    ----------
    hparams : dict
        Keys: num_layers, neurons, lr, activation, dropout
        (as returned by ``ga.chromosome.decode``).
    """
    activation_cls = _ACTIVATION_MAP[hparams["activation"]]
    layers: list[nn.Module] = []
    in_features = config.INPUT_DIM

    for _ in range(hparams["num_layers"]):
        layers.append(nn.Linear(in_features, hparams["neurons"]))
        layers.append(activation_cls())
        layers.append(nn.Dropout(hparams["dropout"]))
        in_features = hparams["neurons"]

    layers.append(nn.Linear(in_features, config.NUM_CLASSES))
    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
