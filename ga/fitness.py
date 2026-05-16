"""Multi-objective fitness evaluation.

Decodes a chromosome, builds a model, trains it for a short number of
epochs, and returns (f1_score, param_count) as the two objectives.
"""

from ga.chromosome import decode
from models.mlp_builder import build_mlp, count_parameters
from engine.trainer import train_and_evaluate


def evaluate(individual, train_loader, val_loader):
    """Evaluate a single individual and return its fitness tuple.

    Returns
    -------
    (f1, param_count) : tuple[float, int]
        Maximise f1, minimise param_count (DEAP weights handle direction).
    """
    hparams = decode(individual)
    model = build_mlp(hparams)

    # Attach lr so the trainer can build the optimizer
    model._lr = hparams["lr"]

    f1, param_count, _ = train_and_evaluate(model, train_loader, val_loader)
    return f1, param_count
