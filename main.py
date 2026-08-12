"""
Main entry point for arithmetic error localization experiment.
"""

import json
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import SEED, DEVICE, MODEL_NAME, POSITIONS, PERCENTILE
from utils import (
    MultiTokenScorer,
    MultiPositionHookedModel,
    generate_problem_pool,
    evaluate_pool,
    build_analog_pairs_stratified,
    position_screen,
    activation_patching_confirmation,
    statistical_treatment,
    run_correction_experiment,
    plot_all_figures,
    run_probing_pipeline,
)


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_study(percentile=PERCENTILE, positions=POSITIONS):
    """Run the complete experimental pipeline."""
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Print header
    print("=" * 80)
    print("MULTI-TOKEN / MULTI-POSITION LOCALIZATION -- REVISION 5")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print(f"Model: {MODEL_NAME}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{model.config.n_layer} layers, {model.config.n_head} heads")
    
    # Initialize scorers
    scorer = MultiTokenScorer(tokenizer, model)
    hooked = MultiPositionHookedModel(model, tokenizer)
    
    # Generate and evaluate problem pool
    pool = generate_problem_pool(seed=SEED)
    pool, corrects, errors = evaluate_pool(tokenizer, scorer, pool, DEVICE)
    
    if len(errors) < 10:
        return {"status": "insufficient_errors", "n_errors": len(errors)}
    
    # Build analog pairs
    discovery_pairs, held_out_pairs = build_analog_pairs_stratified(errors, corrects, scorer)
    print(f"\nAnalog pairs total: {len(discovery_pairs)} discovery, {len(held_out_pairs)} held-out")
    
    if len(discovery_pairs) < 5 or len(held_out_pairs) < 5:
        return {"status": "insufficient_pairs",
                "discovery": len(discovery_pairs), "held_out": len(held_out_pairs)}
    
    # Position screen
    discovery_errors_for_screen = [e for e, _ in discovery_pairs]
    candidates, mean_deg, threshold = position_screen(
        hooked, scorer, discovery_errors_for_screen, list(positions), percentile=percentile
    )
    
    if not candidates:
        return {"status": "no_candidates"}
    
    # Activation patching confirmation
    effects = activation_patching_confirmation(hooked, scorer, discovery_pairs, candidates)
    stats_results = statistical_treatment(effects)
    
    # Get validated components
    validated = [r for r in stats_results if r["significant"]]
    using_fallback = len(validated) == 0
    if using_fallback:
        print("\nWARNING: 0 components survived BH correction. Falling back to top-3 by raw effect.")
    
    targeted = [r["component"] for r in validated] or [r["component"] for r in stats_results[:3]]
    excluded = {(l, c) for l, c, pos, _ in candidates}
    
    # Rank significant components
    ranked_significant = [r["component"] for r in sorted(validated, key=lambda r: r["q_value"])] or targeted
    
    # Run correction experiment
    correction = run_correction_experiment(
        hooked, scorer, held_out_pairs, targeted, excluded, corrects,
        ranked_significant, positions
    )
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"RQ1: {len(validated)} BH-significant of {len(candidates)} screened candidates")
    print(f"RQ2: Targeted components = {targeted}")
    print(f"RQ3: Targeted correction = {correction['targeted_rate']:.2%}")
    print(f"RQ3: Random control     = {correction['random_rate']:.2%}")
    print(f"RQ3: McNemar p-value    = {correction['mcnemar']['pvalue']:.4f}")
    print(f"RQ3: Oracle ceiling     = {correction['oracle_ceiling']['rate']:.2%}")
    
    cd = correction["collateral_damage"]
    if cd["targeted"] and cd["random"]:
        print(f"Collateral: targeted broke {cd['targeted']['rate']:.2%}, "
              f"random broke {cd['random']['rate']:.2%} of previously-correct problems")
    
    # Run probing pipeline
    probing_results = run_probing_pipeline(pool, model, tokenizer, DEVICE)
    if probing_results:
        print(f"\nProbing: best-layer MLP probe AUC = {probing_results['best_auc']:.3f} (Layer {probing_results['best_layer']})")
        print(f"Probing: self-correction precision = {probing_results['self_correction']['precision']:.2%}, "
              f"causal-patch fix rate = {probing_results['self_correction']['fix_rate']:.2%}")
    
    # Generate all figures
    plot_all_figures(correction, stats_results, discovery_pairs, held_out_pairs,
                     errors, corrects, mean_deg, candidates, hooked)
    
    # Return results
    return {
        "status": "ok",
        "n_errors": len(errors),
        "n_correct": len(corrects),
        "n_candidates": len(candidates),
        "n_significant": len(validated),
        "targeted_components": targeted,
        "targeted_is_unvalidated_fallback": using_fallback,
        "correction": correction,
        "probing": probing_results,
    }


if __name__ == "__main__":
    # Run the study
    results = run_study()
    
    # Save results
    with open("/kaggle/working/multi_position_results_v5.json", "w") as f:
        def jsonable(o):
            if isinstance(o, dict):
                return {str(k): jsonable(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [jsonable(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o
        json.dump(jsonable(results), f, indent=2)
    
    print("\nSaved to /kaggle/working/multi_position_results_v5.json")