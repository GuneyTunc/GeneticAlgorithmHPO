"""Chromosome representation for the MO-GA-HPO system.

Each individual is a list of 5 genes:
    [num_layers, neurons, lr, activation_idx, dropout]

Categorical genes (neurons, activation) are stored as indices into their
respective option lists so that all genes are numeric, which simplifies
crossover and mutation.
"""

import math
import random

from deap import base, creator, tools

import config

# ── DEAP fitness & individual types ──────────────────────────────────────────
# weights: maximise F1, minimise param_count
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


# ── Gene generators ──────────────────────────────────────────────────────────
def _rand_num_layers():
    return random.randint(*config.LAYER_RANGE)


def _rand_neuron_idx():
    return random.randint(0, len(config.NEURON_OPTIONS) - 1)


def _rand_lr():
    log_lo = math.log10(config.LR_RANGE[0])
    log_hi = math.log10(config.LR_RANGE[1])
    return 10 ** random.uniform(log_lo, log_hi)


def _rand_activation_idx():
    return random.randint(0, len(config.ACTIVATION_OPTIONS) - 1)


def _rand_dropout():
    return random.uniform(*config.DROPOUT_RANGE)


def random_individual():
    """Return a new random Individual (list of 5 genes)."""
    genes = [
        _rand_num_layers(),
        _rand_neuron_idx(),
        _rand_lr(),
        _rand_activation_idx(),
        _rand_dropout(),
    ]
    return creator.Individual(genes)


# ── Decode helpers ───────────────────────────────────────────────────────────
def decode(individual):
    """Convert raw gene list to a readable dict of hyperparameters."""
    return {
        "num_layers": int(individual[0]),
        "neurons": config.NEURON_OPTIONS[int(individual[1])],
        "lr": individual[2],
        "activation": config.ACTIVATION_OPTIONS[int(individual[3])],
        "dropout": individual[4],
    }
