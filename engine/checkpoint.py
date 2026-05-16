"""Checkpoint utilities for pausing and resuming the evolutionary loop.

Saves the full GA state (population, history, generation counter, RNG state,
elapsed time) to a pickle file so that a run can be interrupted with Ctrl-C
and resumed later from the exact same point.
"""

import os
import pickle
import random
from datetime import datetime

from ga.chromosome import creator  # noqa: F401  — registers DEAP types before unpickling


CHECKPOINT_DIR = os.path.join("results", "checkpoints")
_CHECKPOINT_FILE = "ga_checkpoint.pkl"


def _ensure_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def checkpoint_path() -> str:
    return os.path.join(CHECKPOINT_DIR, _CHECKPOINT_FILE)


def save_checkpoint(
    population,
    generation: int,
    history: dict,
    elapsed: float,
    args_dict: dict,
):
    """Persist the full GA state to disk.

    Parameters
    ----------
    population : list[Individual]
        Current DEAP population (each Individual carries its fitness).
    generation : int
        Last *completed* generation number.
    history : dict
        Per-generation statistics accumulated so far.
    elapsed : float
        Total wall-clock seconds spent in evolution *so far*.
    args_dict : dict
        CLI arguments (pop, gen, seed) so we can validate on resume.
    """
    _ensure_dir()

    pop_data = []
    for ind in population:
        pop_data.append({
            "genes": list(ind),
            "fitness": ind.fitness.values if ind.fitness.valid else None,
        })

    state = {
        "generation": generation,
        "population": pop_data,
        "history": history,
        "random_state": random.getstate(),
        "elapsed": elapsed,
        "args": args_dict,
        "timestamp": datetime.now().isoformat(),
    }

    path = checkpoint_path()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

    print(f"  [Checkpoint] Generation {generation} saved → {path}")


def load_checkpoint():
    """Load a previously saved checkpoint.

    Returns
    -------
    dict  with keys: generation, population (list[Individual]),
          history, elapsed, args, timestamp.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists.
    """
    path = checkpoint_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    with open(path, "rb") as f:
        state = pickle.load(f)

    population = []
    for entry in state["population"]:
        ind = creator.Individual(entry["genes"])
        if entry["fitness"] is not None:
            ind.fitness.values = entry["fitness"]
        population.append(ind)

    random.setstate(state["random_state"])

    return {
        "generation": state["generation"],
        "population": population,
        "history": state["history"],
        "elapsed": state["elapsed"],
        "args": state["args"],
        "timestamp": state["timestamp"],
    }


def has_checkpoint() -> bool:
    return os.path.exists(checkpoint_path())


def delete_checkpoint():
    path = checkpoint_path()
    if os.path.exists(path):
        os.remove(path)
        print(f"  [Checkpoint] Deleted → {path}")
