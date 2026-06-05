import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_DIR = "./dataset"
NUM_CLASSES = 10
INPUT_DIM = 28 * 28  # Fashion-MNIST flattened
BATCH_SIZE = 512
VAL_RATIO = 0.15  # 15% of training set used for validation

# ── Genetic Algorithm ─────────────────────────────────────────────────────────
POPULATION_SIZE = 50
NUM_GENERATIONS = 30
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2
TOURNAMENT_SIZE = 3

# ── Gene Bounds ───────────────────────────────────────────────────────────────
LAYER_RANGE = (1, 5)
NEURON_OPTIONS = [16, 32, 64, 128, 256, 512]
LR_RANGE = (1e-4, 1e-1)
ACTIVATION_OPTIONS = ["relu", "tanh", "elu"]
DROPOUT_RANGE = (0.0, 0.5)

# ── Training (per-individual fitness evaluation) ──────────────────────────────
FITNESS_EPOCHS = 5
