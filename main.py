"""MO-GA-HPO — Multi-Objective Genetic Algorithm for Hyperparameter Optimisation.

Run:
    python main.py                      # full run (defaults from config.py)
    python main.py --pop 10 --gen 3     # quick smoke-test
    python main.py --resume             # resume from last checkpoint
    python main.py --resume --gen 50    # resume and extend to 50 total generations

Press Ctrl-C at any time to pause.  The current state is saved automatically
after every generation, so you can always pick up where you left off.
"""

import argparse
import copy
import random
import time

from deap import base, tools
from tqdm import tqdm

import config
from data.data_loader import get_dataloaders
from ga.chromosome import random_individual, decode
from ga.fitness import evaluate
from ga.operators import uniform_crossover, mutate
from ga.pareto import get_pareto_front, select_champions, extract_front_data
from engine.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    has_checkpoint,
    delete_checkpoint,
)
from analysis.visualizer import (
    plot_pareto_front,
    plot_generation_progress,
    plot_confusion,
    print_champion_summary,
)

_pause_requested = False


def _parse_args():
    p = argparse.ArgumentParser(description="MO-GA-HPO")
    p.add_argument("--pop", type=int, default=config.POPULATION_SIZE)
    p.add_argument("--gen", type=int, default=config.NUM_GENERATIONS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume", action="store_true",
        help="Resume evolution from the last checkpoint.",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    print(f"Device : {config.DEVICE}")
    print(f"Population : {args.pop}")
    print(f"Generations: {args.gen}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── DEAP toolbox ──────────────────────────────────────────────────────
    toolbox = base.Toolbox()
    toolbox.register("individual", random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, train_loader=train_loader, val_loader=val_loader)
    toolbox.register("mate", uniform_crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)

    # ── Resume or fresh start ─────────────────────────────────────────────
    start_gen = 0
    prior_elapsed = 0.0
    history = {"gen": [], "avg_f1": [], "best_f1": [], "avg_params": [], "min_params": []}

    if args.resume and has_checkpoint():
        ckpt = load_checkpoint()
        pop = ckpt["population"]
        start_gen = ckpt["generation"]
        history = ckpt["history"]
        prior_elapsed = ckpt["elapsed"]

        print(f"  Resumed from generation {start_gen}  "
              f"(saved {ckpt['timestamp']})")
        print(f"  Prior elapsed time: {prior_elapsed / 60:.1f} min")

        if args.gen <= start_gen:
            args.gen = max(start_gen + 10, args.gen)
            print(f"  Target generations raised to {args.gen} "
                  f"(was <= checkpoint generation)")
        print()
    else:
        random.seed(args.seed)
        pop = toolbox.population(n=args.pop)

        print("Evaluating initial population …")
        for ind in tqdm(pop, desc="Gen 0", unit="ind"):
            ind.fitness.values = toolbox.evaluate(ind)

        _record_stats(history, 0, pop)

    args_dict = {"pop": args.pop, "gen": args.gen, "seed": args.seed}

    total_start = time.perf_counter()

    # ── Evolutionary loop (with pause support) ────────────────────────────
    global _pause_requested
    _pause_requested = False
    completed_gen = start_gen

    try:
        for gen in range(start_gen + 1, args.gen + 1):
            offspring = toolbox.select(pop, len(pop))
            offspring = [copy.deepcopy(ind) for ind in offspring]

            for i in range(0, len(offspring) - 1, 2):
                if random.random() < config.CROSSOVER_PROB:
                    toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            for ind in offspring:
                if random.random() < config.MUTATION_PROB:
                    toolbox.mutate(ind)
                    del ind.fitness.values

            invalids = [ind for ind in offspring if not ind.fitness.valid]
            for ind in tqdm(invalids, desc=f"Gen {gen}", unit="ind"):
                ind.fitness.values = toolbox.evaluate(ind)

            combined = pop + offspring
            pop = toolbox.select(combined, args.pop)

            _record_stats(history, gen, pop)
            completed_gen = gen

            front = get_pareto_front(pop)
            perf, eff = select_champions(front)
            print(
                f"  Gen {gen:>3d} | Front size: {len(front):>3d} | "
                f"Best F1: {perf.fitness.values[0]:.4f} | "
                f"Min params: {eff.fitness.values[1]:,.0f}"
            )

            elapsed = prior_elapsed + (time.perf_counter() - total_start)
            save_checkpoint(pop, gen, history, elapsed, args_dict)

            if _pause_requested:
                print("\n  Paused by user. Use --resume to continue.\n")
                return

    except KeyboardInterrupt:
        elapsed = prior_elapsed + (time.perf_counter() - total_start)
        print("\n\n  Ctrl-C received — saving checkpoint …")
        save_checkpoint(pop, completed_gen, history, elapsed, args_dict)
        print("  Paused. Run with --resume to continue.\n")
        return

    elapsed = prior_elapsed + (time.perf_counter() - total_start)
    print(f"\nEvolution completed in {elapsed / 60:.1f} minutes.\n")

    # ── Final analysis ────────────────────────────────────────────────────
    front = get_pareto_front(pop)
    perf_champion, eff_champion = select_champions(front)

    print_champion_summary(perf_champion, eff_champion)

    front_data = extract_front_data(front)
    plot_pareto_front(front_data)
    plot_generation_progress(history)
    plot_confusion(perf_champion, eff_champion, val_loader)

    delete_checkpoint()


def _record_stats(history, gen, pop):
    f1s = [ind.fitness.values[0] for ind in pop]
    params = [ind.fitness.values[1] for ind in pop]
    history["gen"].append(gen)
    history["avg_f1"].append(sum(f1s) / len(f1s))
    history["best_f1"].append(max(f1s))
    history["avg_params"].append(sum(params) / len(params))
    history["min_params"].append(min(params))


if __name__ == "__main__":
    main()
