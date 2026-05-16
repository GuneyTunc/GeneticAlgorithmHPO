"""Genetic operators: crossover, mutation, and repair/clamping utilities."""

import math
import random

import config

# Gene indices
_IDX_LAYERS = 0
_IDX_NEURONS = 1
_IDX_LR = 2
_IDX_ACTIVATION = 3
_IDX_DROPOUT = 4


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def repair(individual):
    """Clamp / round genes so they stay within valid bounds."""
    individual[_IDX_LAYERS] = _clamp(
        round(individual[_IDX_LAYERS]), *config.LAYER_RANGE,
    )
    individual[_IDX_NEURONS] = _clamp(
        round(individual[_IDX_NEURONS]), 0, len(config.NEURON_OPTIONS) - 1,
    )
    individual[_IDX_LR] = _clamp(individual[_IDX_LR], *config.LR_RANGE)
    individual[_IDX_ACTIVATION] = _clamp(
        round(individual[_IDX_ACTIVATION]), 0, len(config.ACTIVATION_OPTIONS) - 1,
    )
    individual[_IDX_DROPOUT] = _clamp(individual[_IDX_DROPOUT], *config.DROPOUT_RANGE)
    return individual


def uniform_crossover(ind1, ind2):
    """Swap each gene independently with 50 % probability."""
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    repair(ind1)
    repair(ind2)
    return ind1, ind2


def mutate(individual, indpb=0.3):
    """Mutate each gene independently with probability *indpb*.

    - Integer/index genes: random re-sample within range.
    - Continuous genes: Gaussian perturbation.
    - LR uses log-space perturbation.
    """
    if random.random() < indpb:
        individual[_IDX_LAYERS] = random.randint(*config.LAYER_RANGE)

    if random.random() < indpb:
        individual[_IDX_NEURONS] = random.randint(0, len(config.NEURON_OPTIONS) - 1)

    if random.random() < indpb:
        log_lr = math.log10(individual[_IDX_LR])
        log_lr += random.gauss(0, 0.5)
        log_lo = math.log10(config.LR_RANGE[0])
        log_hi = math.log10(config.LR_RANGE[1])
        individual[_IDX_LR] = 10 ** _clamp(log_lr, log_lo, log_hi)

    if random.random() < indpb:
        individual[_IDX_ACTIVATION] = random.randint(
            0, len(config.ACTIVATION_OPTIONS) - 1,
        )

    if random.random() < indpb:
        individual[_IDX_DROPOUT] += random.gauss(0, 0.1)
        individual[_IDX_DROPOUT] = _clamp(
            individual[_IDX_DROPOUT], *config.DROPOUT_RANGE,
        )

    repair(individual)
    return (individual,)
