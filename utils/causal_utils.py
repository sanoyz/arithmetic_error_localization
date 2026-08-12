"""
Causal utilities: zero-ablation, activation patching, correction experiment.
"""

import random
import numpy as np

from config import SEED, N_NULL, PERCENTILE, POSITIONS, COLLATERAL_N_SAMPLE, ORACLE_N_SAMPLE


def position_screen(hooked, scorer, discovery_errors, positions, n_null=N_NULL, 
                    percentile=PERCENTILE, seed=SEED):
    """Zero-ablation screen to identify candidate components."""
    print("\n" + "=" * 80)
    print("ZERO-ABLATION SCREEN ACROSS POSITIONS (heads + MLPs)")
    print("=" * 80)
    
    all_comp = hooked.all_components()
    print(f"Components: {len(all_comp)} ({hooked.n_layers * hooked.n_heads} heads + "
          f"{hooked.n_layers} MLPs), positions tested: {positions}, "
          f"discovery errors: {len(discovery_errors)}")

    degradation = {(l, c, pos): [] for (l, c) in all_comp for pos in positions}
    
    for idx, p in enumerate(discovery_errors):
        if idx % 10 == 0:
            print(f"  Processing error {idx+1}/{len(discovery_errors)}...")
        base_p = scorer.sequence_probability(p["input_ids"], p["answer_tokens"])
        for (l, c) in all_comp:
            for pos in positions:
                ap = hooked.ablated_prob(p["input_ids"], p["answer_tokens"], scorer, l, c, pos)
                degradation[(l, c, pos)].append(base_p - ap)

    mean_degradation = {k: float(np.mean(v)) for k, v in degradation.items()}

    best_per_component = {}
    for (l, c, pos), deg in mean_degradation.items():
        key = (l, c)
        if key not in best_per_component or deg > best_per_component[key][1]:
            best_per_component[key] = (pos, deg)

    ranked = sorted(best_per_component.items(), key=lambda kv: -kv[1][1])

    n_exclude = max(1, int(len(best_per_component) * 0.15))
    top_pool = {c for c, _ in ranked[:n_exclude]}
    remaining = [c for c in best_per_component if c not in top_pool]
    rng = random.Random(seed)
    null_components = rng.sample(remaining, min(n_null, len(remaining)))
    null_means = [best_per_component[c][1] for c in null_components]

    threshold = float(np.percentile(null_means, percentile))
    candidates = [(l, c, pos, deg) for (l, c), (pos, deg) in best_per_component.items()
                  if deg > threshold]
    candidates.sort(key=lambda x: -x[3])

    print(f"\nNull distribution ({len(null_means)} per-component means), "
          f"{percentile}th percentile: {threshold:.6f}")
    print(f"Candidates found: {len(candidates)} of {len(best_per_component)} components")
    for l, c, pos, deg in candidates[:15]:
        label = f"MLP" if c == "mlp" else f"Head {c}"
        print(f"  Layer {l}, {label}, Position {pos}: {deg:.6f}")

    return candidates, mean_degradation, threshold


def activation_patching_confirmation(hooked, scorer, discovery_pairs, candidates):
    """Confirm candidates via activation patching."""
    print("\n" + "=" * 80)
    print("CONFIRMATORY ACTIVATION PATCHING")
    print("=" * 80)
    
    comps = [(l, c) for l, c, pos, _ in candidates]
    poss = [pos for l, c, pos, _ in candidates]

    effects = {(l, c, pos): [] for l, c, pos, _ in candidates}
    for idx, (err, analog) in enumerate(discovery_pairs):
        if idx % 10 == 0:
            print(f"  Processing pair {idx+1}/{len(discovery_pairs)}...")
        cache = hooked.cache_components(analog["input_ids"], comps, poss)
        corrupted_p = scorer.sequence_probability(err["input_ids"], err["answer_tokens"])
        for (l, c), pos in zip(comps, poss):
            patched_p = hooked.patched_prob(
                err["input_ids"], err["answer_tokens"], scorer, [(l, c)], [pos], cache
            )
            effects[(l, c, pos)].append(patched_p - corrupted_p)
    return effects


def collateral_damage_check(hooked, scorer, comps, poss, holdout_correct_pool,
                             label, n_sample=COLLATERAL_N_SAMPLE, seed=SEED):
    """Check if patching components breaks previously-correct problems."""
    rng = random.Random(seed)
    sample = rng.sample(holdout_correct_pool, min(n_sample, len(holdout_correct_pool)))
    if not sample:
        print(f"  {label}: no held-out correct problems available for collateral check")
        return None

    n_broken = 0
    for p in sample:
        cache = hooked.cache_components(p["input_ids"], comps, poss)
        still_matches = hooked.patched_matches(
            p["input_ids"], p["answer_tokens"], comps, poss, cache
        )
        if not still_matches:
            n_broken += 1
    rate = n_broken / len(sample)
    print(f"  {label}: {rate:.2%} of previously-correct problems broken by patching "
          f"({n_broken}/{len(sample)})")
    return {"label": label, "n_sample": len(sample), "n_broken": n_broken, "rate": rate}


def full_circuit_ceiling(hooked, scorer, held_out_pairs, positions, n_sample=ORACLE_N_SAMPLE, seed=SEED):
    """Oracle ceiling: patch every component at every position simultaneously."""
    print("\n" + "=" * 80)
    print("FULL-CIRCUIT ORACLE CEILING (all components, all tested positions)")
    print("=" * 80)
    
    rng = random.Random(seed)
    sample = rng.sample(held_out_pairs, min(n_sample, len(held_out_pairs)))
    all_comp = hooked.all_components()
    comps, poss = [], []
    for (l, c) in all_comp:
        for pos in positions:
            comps.append((l, c))
            poss.append(pos)

    n_fixed = 0
    for err, analog in sample:
        cache = hooked.cache_components(analog["input_ids"], comps, poss)
        fixed = hooked.patched_matches(err["input_ids"], err["answer_tokens"], comps, poss, cache)
        n_fixed += int(fixed)
    rate = n_fixed / len(sample) if sample else 0.0
    print(f"  Oracle ceiling: {rate:.2%} ({n_fixed}/{len(sample)}) fixed when "
          f"EVERY component at EVERY tested position is patched from the analog.")
    print("  Interpretation: this is the ceiling on what activation patching at "
          "these positions can achieve at all.")
    return {"rate": rate, "n_fixed": n_fixed, "n_sample": len(sample)}


def greedy_circuit_curve(hooked, scorer, held_out_pairs, ranked_significant, seed=SEED):
    """Grow the circuit one component at a time and track correction rate."""
    print("\n" + "=" * 80)
    print("GREEDY CIRCUIT-GROWING CURVE")
    print("=" * 80)
    
    if not ranked_significant:
        print("  No significant components to grow a circuit from.")
        return []

    results = []
    for k in range(1, len(ranked_significant) + 1):
        subset = ranked_significant[:k]
        comps = [(l, c) for l, c, pos in subset]
        poss = [pos for l, c, pos in subset]
        n_fixed = 0
        for err, analog in held_out_pairs:
            cache = hooked.cache_components(analog["input_ids"], comps, poss)
            fixed = hooked.patched_matches(err["input_ids"], err["answer_tokens"], comps, poss, cache)
            n_fixed += int(fixed)
        rate = n_fixed / len(held_out_pairs) if held_out_pairs else 0.0
        print(f"  k={k}: {rate:.2%} ({n_fixed}/{len(held_out_pairs)})")
        results.append({"k": k, "rate": rate, "n_fixed": n_fixed})
    return results


def run_correction_experiment(hooked, scorer, held_out_pairs, targeted, excluded,
                               correct_pool_for_collateral, ranked_significant,
                               positions, seed=SEED):
    """Run the full correction experiment with targeted and random controls."""
    print("\n" + "=" * 80)
    print("CORRECTION EXPERIMENT")
    print("=" * 80)

    MIN_HELD_OUT_FOR_20PT_EFFECT = 63
    if len(held_out_pairs) < MIN_HELD_OUT_FOR_20PT_EFFECT:
        print(f"WARNING: {len(held_out_pairs)} held-out pairs is below the "
              f"~{MIN_HELD_OUT_FOR_20PT_EFFECT} needed for 80% power.")

    def score(pairs, comps, poss, label):
        n_fixed = 0
        for err, analog in pairs:
            cache = hooked.cache_components(analog["input_ids"], comps, poss)
            fixed = hooked.patched_matches(err["input_ids"], err["answer_tokens"], comps, poss, cache)
            n_fixed += int(fixed)
        rate = n_fixed / len(pairs) if pairs else 0.0
        print(f"  {label}: {rate:.2%} ({n_fixed}/{len(pairs)})")
        return rate, n_fixed

    t_comps = [(l, c) for l, c, pos in targeted]
    t_poss = [pos for l, c, pos in targeted]
    targeted_rate, _ = score(held_out_pairs, t_comps, t_poss, f"Targeted ({len(t_comps)} comp), all ops")

    rng = random.Random(seed)
    all_comp = hooked.all_components()
    non_candidates = [c for c in all_comp if c not in excluded]
    random_comps = rng.sample(non_candidates, min(len(t_comps), len(non_candidates)))
    random_poss = [rng.choice(list(positions)) for _ in random_comps]
    random_rate, _ = score(held_out_pairs, random_comps, random_poss,
                            f"Random control ({len(random_comps)} comp), all ops")

    # McNemar's test
    from .stats_utils import mcnemar_test
    mcnemar_result = mcnemar_test(held_out_pairs, hooked, scorer, t_comps, t_poss,
                                   random_comps, random_poss)

    # Per-operation breakdown
    MIN_HELD_OUT_PER_OP = 5
    by_op = {}
    print(f"\n  Per-operation breakdown (RQ4; stratum reported only if "
          f">= {MIN_HELD_OUT_PER_OP} held-out pairs):")
    for op in ("add", "sub", "mul"):
        op_pairs = [(e, a) for e, a in held_out_pairs if e["op"] == op]
        if len(op_pairs) < MIN_HELD_OUT_PER_OP:
            print(f"    {op}: only {len(op_pairs)} held-out pairs -- INSUFFICIENT")
            by_op[op] = {"n": len(op_pairs), "status": "insufficient"}
            continue
        t_rate, t_fixed = score(op_pairs, t_comps, t_poss, f"    {op} targeted")
        r_rate, r_fixed = score(op_pairs, random_comps, random_poss, f"    {op} random ")
        by_op[op] = {
            "n": len(op_pairs), "status": "reported",
            "targeted_rate": t_rate, "targeted_fixed": t_fixed,
            "random_rate": r_rate, "random_fixed": r_fixed,
        }

    # Collateral damage
    print("\n  Collateral damage check:")
    used_as_analog = {id(a) for _, a in held_out_pairs}
    spare_correct = [p for p in correct_pool_for_collateral if id(p) not in used_as_analog]
    collateral_targeted = collateral_damage_check(
        hooked, scorer, t_comps, t_poss, spare_correct, "Targeted components"
    )
    collateral_random = collateral_damage_check(
        hooked, scorer, random_comps, random_poss, spare_correct, "Random control"
    )

    # Oracle ceiling and greedy curve
    ceiling = full_circuit_ceiling(hooked, scorer, held_out_pairs, positions)
    curve = greedy_circuit_curve(hooked, scorer, held_out_pairs, ranked_significant)

    return {
        "targeted_rate": targeted_rate, "random_rate": random_rate,
        "improvement": targeted_rate - random_rate,
        "n_held_out": len(held_out_pairs),
        "underpowered": len(held_out_pairs) < MIN_HELD_OUT_FOR_20PT_EFFECT,
        "mcnemar": mcnemar_result,
        "by_operation": by_op,
        "collateral_damage": {"targeted": collateral_targeted, "random": collateral_random},
        "oracle_ceiling": ceiling,
        "greedy_curve": curve,
    }