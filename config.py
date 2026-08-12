"""
Configuration file for arithmetic error localization experiment.
"""

import torch

# ============================================================================
# Global Configuration
# ============================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_NAME = "FlameF0X/MathGPT2"
N_LAYERS = 6
N_HEADS = 12
HEAD_DIM = 64  # 768 / 12

# ============================================================================
# Data Configuration
# ============================================================================

HARD_RANGES = {
    "add": (50, 250),
    "sub": (50, 700),
    "mul": (10, 99),
}

EASY_N = 80
HARD_N = 400

# ============================================================================
# Experiment Configuration
# ============================================================================

POSITIONS = (0, 1, 2, 3, 4)
PERCENTILE = 85
N_NULL = 20
ALPHA = 0.05
DISCOVERY_FRAC = 0.7

# ============================================================================
# Correction Experiment Configuration
# ============================================================================

MIN_HELD_OUT_PER_OP = 5
MIN_HELD_OUT_FOR_20PT_EFFECT = 63
COLLATERAL_N_SAMPLE = 40
ORACLE_N_SAMPLE = 30

# ============================================================================
# Analog Matching Configuration
# ============================================================================

BASE_TOLERANCE = 50
TOLERANCE_FRAC = 0.08
REQUIRE_SAME_PROMPT_LENGTH = True

# ============================================================================
# Probing Configuration
# ============================================================================

PROBE_TRAIN_SPLIT = 0.7
PROBE_VAL_SPLIT = 0.15
PROBE_TEST_SPLIT = 0.15
PROBE_MLP_HIDDEN_DIM = 64
PROBE_L2_REGULARIZATION = 0.001
PROBE_EPOCHS = 100
PROBE_BATCH_SIZE = 32
PROBE_LEARNING_RATE = 0.001

# ============================================================================
# Visualization Configuration
# ============================================================================

FIGURE_DPI = 300
FIGURE_FORMAT = "png"
COLORS = {
    "targeted": "#2E86AB",
    "random": "#A23B72",
    "collateral": "#D62828",
    "threshold": "#F18F01",
    "significant": "#D62828",
    "non_significant": "#808080",
}