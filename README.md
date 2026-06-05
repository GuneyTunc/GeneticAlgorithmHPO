# MO-GA-HPO — Multi-Objective Genetic Algorithm for Hyperparameter Optimization

A multi-objective hyperparameter optimization framework that uses a **genetic algorithm (NSGA-II)** to discover neural network configurations that balance **classification performance** against **model size**, instead of optimizing accuracy alone.

Rather than producing a single "best" model, the system evolves a **Pareto front** of trade-off solutions and reports two champions:

- **Performance Champion** — highest validation F1-score
- **Efficiency Champion** — smallest parameter count

The project follows a "Green AI / Efficiency-Aware Machine Learning" philosophy: find models that are not only accurate, but also resource-efficient.

---

## Key Idea

Traditional hyperparameter optimization minimizes a single error metric. This project solves a **trade-off problem** with two competing objectives:

| Objective | Goal | Metric |
|-----------|------|--------|
| f₁ | Maximize | Macro F1-score (classification quality) |
| f₂ | Minimize | Trainable parameter count (model complexity) |

NSGA-II (non-dominated sorting + crowding distance) optimizes both simultaneously, so no manual scalar weighting between the objectives is required.

---

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research, 2017) — a 10-class grayscale image classification benchmark.

| Property | Value |
|----------|-------|
| Image size | 28×28 grayscale (flattened to 784) |
| Classes | 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |
| Train | 60,000 (split 85% train / 15% validation, `seed=42`) |
| Test | 10,000 (held out, used only for final evaluation) |

**Preprocessing:** `ToTensor()` → [0, 1] scaling, `Normalize((0.5,), (0.5,))` → ≈[-1, 1], and flattening to a 784-dim vector at training time. No data augmentation is applied. The dataset is downloaded automatically on first run.

---

## Chromosome (Search Space)

Each individual encodes one MLP as a list of 5 genes:

| Gene | Type | Range / Options |
|------|------|-----------------|
| Number of layers | Integer | 1 – 5 |
| Neurons per layer | Categorical | {16, 32, 64, 128, 256, 512} |
| Learning rate | Continuous (log-scale) | 1e-4 – 1e-1 |
| Activation | Categorical | ReLU, Tanh, ELU |
| Dropout | Continuous | 0.0 – 0.5 |

---

## Evolutionary Loop

```
Gen 0:  50 random chromosomes → train each 5 epochs → (F1, param_count)
   │
Each generation:
   1. Select 50 parents with NSGA-II
   2. Copy parents → offspring
   3. Uniform crossover (per pair, p = 0.70)
   4. Mutation (per individual, p = 0.20; per gene p = 0.30)
   5. Re-evaluate individuals whose fitness was invalidated
   6. Elitist replacement: combine parents + offspring (100),
      keep best 50 via NSGA-II
   │
After 30 generations: Pareto front → Performance & Efficiency champions
```

---

## Pause / Resume

Long runs can be interrupted at any time with **Ctrl-C**. The full GA state (population, fitness, generation counter, RNG state, elapsed time) is checkpointed automatically after every generation to `results/checkpoints/ga_checkpoint.pkl`.

```bash
python main.py --resume              # continue from the last checkpoint
python main.py --resume --gen 50     # resume and extend to 50 total generations
```

The checkpoint is deleted automatically once evolution finishes successfully.

---

## Project Structure

```
genetic_hpo_project/
├── main.py                      # Entry point: evolutionary loop + checkpointing
├── config.py                    # All constants (GA, data, training)
│
├── data/
│   └── data_loader.py           # Fashion-MNIST download, preprocessing, DataLoaders
├── models/
│   └── mlp_builder.py           # Builds an MLP from decoded hyperparameters
├── engine/
│   ├── trainer.py               # Short training + validation F1 + param count
│   └── checkpoint.py            # Pause/resume state serialization
├── ga/
│   ├── chromosome.py            # DEAP individual/fitness types, decode()
│   ├── operators.py             # Uniform crossover, mutation, repair
│   ├── fitness.py               # individual → model → train → (F1, params)
│   └── pareto.py                # Pareto front + champion selection
├── analysis/
│   ├── visualizer.py            # Pareto plot, generation progress, confusion matrices
│   └── evaluate_champions.py    # Full success + error metrics on the test set
└── results/                     # Generated plots, metrics JSON, reports
```

---

## Installation

Requires Python 3.10+. A CUDA-capable GPU is optional but recommended.

```bash
git clone https://github.com/GuneyTunc/GeneticAlgorithmHPO.git
cd GeneticAlgorithmHPO
pip install -r requirements.txt
```

For GPU acceleration, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Usage

```bash
# Full run (defaults from config.py: 50 population, 30 generations)
python main.py

# Quick smoke test
python main.py --pop 10 --gen 3

# Resume an interrupted run
python main.py --resume

# Evaluate the champions with full metrics on the test set
python analysis/evaluate_champions.py
```

### Command-line arguments (`main.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--pop` | 50 | Population size |
| `--gen` | 30 | Number of generations |
| `--seed` | 42 | Random seed |
| `--resume` | off | Resume from the last checkpoint |

---

## Outputs

After a run, the following are written to `results/`:

- `pareto_front.png` — F1-score vs. parameter count trade-off curve
- `generation_progress.png` — best/avg F1 and model size across generations
- `confusion_matrices.png` — side-by-side confusion matrices of both champions
- `champion_metrics.json` — full success and error metrics (validation + test)
- `champion_error_metrics.md` — formatted metric tables

---

## Configuration

All key settings live in `config.py`:

```python
POPULATION_SIZE = 50      # individuals per generation
NUM_GENERATIONS = 30      # evolutionary iterations
CROSSOVER_PROB  = 0.7     # crossover probability per pair
MUTATION_PROB   = 0.2     # mutation probability per individual
FITNESS_EPOCHS  = 5       # epochs used to evaluate each individual
BATCH_SIZE      = 512
VAL_RATIO       = 0.15
```

---

## Methodological Reference

This work follows the methodology surveyed in:

> Morales-Hernández, A., Van Nieuwenhuyse, I., & Rojas Gonzalez, S. (2023). *A survey on multi-objective hyperparameter optimization algorithms for machine learning.* Artificial Intelligence Review. DOI: [10.1007/s10462-022-10359-2](https://doi.org/10.1007/s10462-022-10359-2)

Core algorithm:

> Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II.* IEEE Transactions on Evolutionary Computation.

---

## Tech Stack

- **PyTorch** — model definition and training
- **DEAP** — genetic algorithm framework (NSGA-II)
- **scikit-learn** — evaluation metrics
- **matplotlib / seaborn** — visualization

---

## Author

**Güney Tunç** — 232932021
Computer Engineering, Artificial Intelligence Course Project
