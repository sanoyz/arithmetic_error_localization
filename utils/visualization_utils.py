"""
Visualization utilities for paper figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import COLORS, POSITIONS


def plot_prob_performance_summary(correction, stats_results):
    """Figure 1: Prob performance summary."""
    print("\n" + "=" * 80)
    print("FIGURE 1: Prob Performance Summary")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: BH correction results summary
    ax = axes[0]
    comp_labels = [f"{c[0]},{c[1]},{c[2]}" for c in [r['component'] for r in stats_results]]
    effects = [r['mean_effect'] for r in stats_results]
    sig = [r['significant'] for r in stats_results]
    
    colors = ['red' if s else 'gray' for s in sig]
    bars = ax.barh(comp_labels, effects, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Patching Effect')
    ax.set_title('Component Effects (BH-significant in red)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy summary
    ax = axes[1]
    metrics = ['Correct Rate', 'Error Rate', 'Targeted Fix', 'Random Fix']
    values = [
        388/1411 * 100,  # Correct rate
        1023/1411 * 100,  # Error rate
        correction['targeted_rate'] * 100,
        correction['random_rate'] * 100
    ]
    colors_list = ['green', 'red', 'blue', 'gray']
    bars = ax.bar(metrics, values, color=colors_list, alpha=0.7)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Performance Summary')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Prob Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_experimental_design(discovery_pairs, held_out_pairs, errors, corrects):
    """Figure 2: Experimental design."""
    print("\n" + "=" * 80)
    print("FIGURE 2: Experimental Design")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Data split
    ax = axes[0]
    discovery_size = len(discovery_pairs)
    held_out_size = len(held_out_pairs)
    unmatched = len(errors) - discovery_size - held_out_size
    
    labels = ['Discovery', 'Held-Out', 'Unmatched']
    sizes = [discovery_size, held_out_size, unmatched]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title(f'Data Split (Total Errors: {len(errors)})')
    
    # Plot 2: By operation
    ax = axes[1]
    op_counts = {}
    for err, _ in discovery_pairs + held_out_pairs:
        op = err['op']
        op_counts[op] = op_counts.get(op, 0) + 1
    
    ops = list(op_counts.keys())
    counts = list(op_counts.values())
    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(ops)]
    bars = ax.bar(ops, counts, color=colors)
    ax.set_xlabel('Operation')
    ax.set_ylabel('Number of Pairs')
    ax.set_title('By Operation')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    # Plot 3: By difficulty
    ax = axes[2]
    difficulty_counts = {'Easy': 0, 'Hard': 0}
    for err, _ in discovery_pairs + held_out_pairs:
        difficulty_counts[err['difficulty']] += 1
    
    labels = list(difficulty_counts.keys())
    sizes = list(difficulty_counts.values())
    colors = ['#2E86AB', '#A23B72']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('By Difficulty')
    
    plt.suptitle('Experimental Design', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_component_heatmaps(mean_degradation, candidates, hooked, positions=POSITIONS):
    """Figure 3: Component discovery heatmaps."""
    print("\n" + "=" * 80)
    print("FIGURE 3: Component Discovery Heatmaps")
    print("=" * 80)
    
    n_layers = hooked.n_layers
    
    # Build attention head matrix
    head_matrix = np.zeros((n_layers, len(positions)))
    for (layer, comp, pos), deg in mean_degradation.items():
        if isinstance(comp, int):
            pos_idx = positions.index(pos)
            if deg > head_matrix[layer, pos_idx]:
                head_matrix[layer, pos_idx] = deg
    
    # Build MLP matrix
    mlp_matrix = np.zeros((n_layers, len(positions)))
    candidate_set = {(l, c, p) for l, c, p, _ in candidates}
    mlp_candidate_mask = np.zeros((n_layers, len(positions)))
    
    for (layer, comp, pos), deg in mean_degradation.items():
        if comp == "mlp":
            pos_idx = positions.index(pos)
            mlp_matrix[layer, pos_idx] = deg
            if (layer, comp, pos) in candidate_set:
                mlp_candidate_mask[layer, pos_idx] = 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap 1: Attention heads
    ax = axes[0]
    sns.heatmap(head_matrix, annot=True, fmt='.4f', ax=ax,
                xticklabels=positions, yticklabels=[f'Layer {i}' for i in range(n_layers)],
                cmap='Reds', vmin=0, vmax=max(head_matrix.max(), 0.01))
    ax.set_title('Attention Heads (Best per Layer-Position)')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    
    # Heatmap 2: MLP modules
    ax = axes[1]
    sns.heatmap(mlp_matrix, annot=True, fmt='.4f', ax=ax,
                xticklabels=positions, yticklabels=[f'Layer {i}' for i in range(n_layers)],
                cmap='Blues', vmin=0, vmax=max(mlp_matrix.max(), 0.01))
    # Overlay candidate markers
    for i in range(n_layers):
        for j, pos in enumerate(positions):
            if mlp_candidate_mask[i, j] == 1:
                ax.text(j + 0.5, i + 0.5, '★', ha='center', va='center', 
                       color='red', fontsize=12, fontweight='bold')
    ax.set_title('MLP Modules (★ = Candidate)')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    
    plt.suptitle('Component Discovery: Layer × Position Heatmaps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"Attention head max: {head_matrix.max():.4f}")
    print(f"MLP max: {mlp_matrix.max():.4f}")
    print(f"MLP/Attention ratio: {mlp_matrix.max() / head_matrix.max():.2f}x")


def plot_correction_results(correction, stats_results):
    """Figure 4: Correction results."""
    print("\n" + "=" * 80)
    print("FIGURE 4: Correction Results")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Targeted vs Random
    ax = axes[0, 0]
    labels = ['Targeted', 'Random']
    rates = [correction['targeted_rate'] * 100, correction['random_rate'] * 100]
    bars = ax.bar(labels, rates, color=['#2E86AB', '#A23B72'], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylabel('Correction Rate (%)')
    ax.set_title(f'Correction Rates (McNemar p={correction["mcnemar"]["pvalue"]:.4f})')
    ax.grid(True, alpha=0.3)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{rate:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: McNemar confusion matrix
    ax = axes[0, 1]
    table = correction['mcnemar']['table']
    sns.heatmap(table, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Random Fixed', 'Random Broke'],
                yticklabels=['Targeted Fixed', 'Targeted Broke'])
    ax.set_title(f'McNemar\'s Test\np={correction["mcnemar"]["pvalue"]:.4f}')
    ax.set_ylabel('Targeted')
    ax.set_xlabel('Random')
    
    # Plot 3: Per-operation breakdown
    ax = axes[1, 0]
    ops = []
    t_rates = []
    r_rates = []
    for op in ('add', 'sub', 'mul'):
        if op in correction['by_operation'] and correction['by_operation'][op]['status'] == 'reported':
            ops.append(op)
            t_rates.append(correction['by_operation'][op]['targeted_rate'] * 100)
            r_rates.append(correction['by_operation'][op]['random_rate'] * 100)
    
    if ops:
        x = np.arange(len(ops))
        width = 0.35
        bars1 = ax.bar(x - width/2, t_rates, width, label='Targeted', color='#2E86AB', alpha=0.7)
        bars2 = ax.bar(x + width/2, r_rates, width, label='Random', color='#A23B72', alpha=0.7)
        ax.set_xlabel('Operation')
        ax.set_ylabel('Correction Rate (%)')
        ax.set_title('Per-Operation Correction Rates')
        ax.set_xticks(x)
        ax.set_xticklabels(ops)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 4: Collateral damage
    ax = axes[1, 1]
    cd = correction['collateral_damage']
    collateral_data = []
    collateral_labels = []
    if cd['targeted']:
        collateral_data.append(cd['targeted']['rate'] * 100)
        collateral_labels.append('Targeted')
    if cd['random']:
        collateral_data.append(cd['random']['rate'] * 100)
        collateral_labels.append('Random')
    
    if collateral_data:
        bars = ax.bar(collateral_labels, collateral_data, color=['#D62828', '#F77F00'], alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('Previously-Correct Broken (%)')
        ax.set_title('Collateral Damage')
        ax.grid(True, alpha=0.3)
        for bar, val in zip(bars, collateral_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Correction Experiment Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_self_correction_outcomes(correction):
    """Figure 5: Self-correction outcomes."""
    print("\n" + "=" * 80)
    print("FIGURE 5: Self-Correction Outcomes")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Greedy circuit-growing curve
    ax = axes[0]
    curve = correction['greedy_curve']
    if curve:
        ks = [r['k'] for r in curve]
        rates = [r['rate'] * 100 for r in curve]
        ax.plot(ks, rates, 'b-o', linewidth=2, markersize=8)
        ax.axhline(y=correction['random_rate'] * 100, color='red', linestyle='--', 
                   label=f'Random Baseline ({correction["random_rate"]:.2%})')
        ax.set_xlabel('Number of Components Patched (k)')
        ax.set_ylabel('Correction Rate (%)')
        ax.set_title('Greedy Circuit-Growing Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Oracle ceiling
    ax = axes[1]
    ceiling = correction['oracle_ceiling']
    categories = ['Oracle Ceiling']
    values = [ceiling['rate'] * 100]
    bars = ax.bar(categories, values, color='#D62828', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylabel('Correction Rate (%)')
    ax.set_title(f'Oracle Ceiling: {ceiling["rate"]:.2%} ({ceiling["n_fixed"]}/{ceiling["n_sample"]})')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Self-Correction Outcomes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_all_figures(correction, stats_results, discovery_pairs, held_out_pairs, 
                     errors, corrects, mean_degradation, candidates, hooked):
    """Generate all paper figures."""
    print("\n" + "=" * 80)
    print("GENERATING PAPER FIGURES")
    print("=" * 80)
    
    plot_prob_performance_summary(correction, stats_results)
    plot_experimental_design(discovery_pairs, held_out_pairs, errors, corrects)
    plot_component_heatmaps(mean_degradation, candidates, hooked)
    plot_correction_results(correction, stats_results)
    plot_self_correction_outcomes(correction)