"""Pareto dominance utilities and front tracking."""

from deap.tools import sortNondominated

from ga.chromosome import decode


def get_pareto_front(population):
    """Return the first (best) Pareto front from *population*.

    Uses DEAP's built-in non-dominated sorting which respects the
    fitness weights defined on the individuals.
    """
    fronts = sortNondominated(population, len(population), first_front_only=True)
    return fronts[0]


def extract_front_data(front):
    """Return a list of dicts with hyperparams + fitness for each member."""
    records = []
    for ind in front:
        hparams = decode(ind)
        records.append({
            **hparams,
            "f1_score": ind.fitness.values[0],
            "param_count": ind.fitness.values[1],
        })
    return records


def select_champions(front):
    """Pick the performance champion and the efficiency champion.

    Returns
    -------
    perf_champion, eff_champion : (individual, individual)
    """
    perf_champion = max(front, key=lambda ind: ind.fitness.values[0])
    eff_champion = min(front, key=lambda ind: ind.fitness.values[1])
    return perf_champion, eff_champion
