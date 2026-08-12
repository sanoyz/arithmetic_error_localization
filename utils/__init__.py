"""
Utility modules for arithmetic error localization.
"""

from .model_utils import MultiTokenScorer, MultiPositionHookedModel
from .data_utils import (
    generate_problem_pool,
    evaluate_pool,
    find_analog,
    build_analog_pairs_stratified,
    HARD_RANGES,
)
from .causal_utils import (
    position_screen,
    activation_patching_confirmation,
    collateral_damage_check,
    full_circuit_ceiling,
    greedy_circuit_curve,
    run_correction_experiment,
)
from .stats_utils import statistical_treatment, mcnemar_test
from .probing_utils import run_probing_pipeline
from .visualization_utils import (
    plot_prob_performance_summary,
    plot_experimental_design,
    plot_component_heatmaps,
    plot_correction_results,
    plot_self_correction_outcomes,
    plot_all_figures,
)

__all__ = [
    "MultiTokenScorer",
    "MultiPositionHookedModel",
    "generate_problem_pool",
    "evaluate_pool",
    "find_analog",
    "build_analog_pairs_stratified",
    "HARD_RANGES",
    "position_screen",
    "activation_patching_confirmation",
    "collateral_damage_check",
    "full_circuit_ceiling",
    "greedy_circuit_curve",
    "run_correction_experiment",
    "statistical_treatment",
    "mcnemar_test",
    "run_probing_pipeline",
    "plot_prob_performance_summary",
    "plot_experimental_design",
    "plot_component_heatmaps",
    "plot_correction_results",
    "plot_self_correction_outcomes",
    "plot_all_figures",
]