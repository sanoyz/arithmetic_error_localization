"""
Statistical utilities: Benjamini-Hochberg correction and McNemar's test.
"""

import numpy as np
from scipy import stats as sstats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar


def statistical_treatment(effects, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction to patching effects.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL TREATMENT (Benjamini-Hochberg)")
    print("=" * 80)
    
    keys = list(effects.keys())
    n_samples = len(next(iter(effects.values()))) if effects else 0
    if n_samples < 8:
        print(f"WARNING: only {n_samples} discovery pairs -- t-tests will have low power.")

    pvals, means = [], []
    for k in keys:
        vals = np.array(effects[k])
        if len(vals) < 2 or np.allclose(vals, vals[0]):
            pvals.append(1.0)
        else:
            _, p = sstats.ttest_1samp(vals, 0.0)
            pvals.append(float(p))
        means.append(float(vals.mean()))

    rejected, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    results = [
        {"component": k, "mean_effect": m, "q_value": float(q), "significant": bool(r)}
        for k, m, q, r in zip(keys, means, qvals, rejected)
    ]
    results.sort(key=lambda d: -d["mean_effect"])
    
    n_sig = sum(r["significant"] for r in results)
    print(f"Significant (component, position) pairs: {n_sig} of {len(results)}")
    for r in results[:15]:
        sig = "*" if r["significant"] else " "
        print(f"  [{sig}] {r['component']}: effect={r['mean_effect']:.4f}, q={r['q_value']:.4f}")
    
    return results


def mcnemar_test(held_out_pairs, hooked, scorer, t_comps, t_poss, r_comps, r_poss):
    """
    Run McNemar's paired test on targeted vs random correction outcomes.
    """
    both_fixed = both_wrong = only_targeted = only_random = 0
    
    for err, analog in held_out_pairs:
        cache_t = hooked.cache_components(analog["input_ids"], t_comps, t_poss)
        fixed_t = hooked.patched_matches(err["input_ids"], err["answer_tokens"], t_comps, t_poss, cache_t)
        
        cache_r = hooked.cache_components(analog["input_ids"], r_comps, r_poss)
        fixed_r = hooked.patched_matches(err["input_ids"], err["answer_tokens"], r_comps, r_poss, cache_r)
        
        if fixed_t and fixed_r:
            both_fixed += 1
        elif fixed_t and not fixed_r:
            only_targeted += 1
        elif fixed_r and not fixed_t:
            only_random += 1
        else:
            both_wrong += 1

    table = [[both_fixed, only_targeted], [only_random, both_wrong]]
    result = mcnemar(table, exact=True)
    
    print(f"\n  McNemar's test (paired): both-fixed={both_fixed}, "
          f"targeted-only={only_targeted}, random-only={only_random}, both-wrong={both_wrong}")
    print(f"  p-value = {result.pvalue:.4f} "
          f"({'significant difference' if result.pvalue < 0.05 else 'NOT significantly different'})")
    
    return {"table": table, "pvalue": float(result.pvalue)}