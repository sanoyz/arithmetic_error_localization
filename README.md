# Where Do Transformers Go Wrong?

**Causal Localization and Correction of Arithmetic Errors in MathGPT-2 (81.9M)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What this does

We localize causal components responsible for *arithmetic errors* in a small transformer and test if activation patching from a correct analog can repair them. 

**The core result**: Identifying *where* a computation happens, and even detecting *in advance* that it will fail (Probe $\text{AUC} = 0.930$), does **not** translate into a working fix ($0.00\%$ correction rate).

---

## Key findings at a glance

- **Localization**: Arithmetic computation is strongly localized to late-layer MLPs. Their causal effect ($\Delta \approx 0.091$) is $\sim 10\times$ larger than any attention head.
- **Correction fails**: Targeted analog patching is statistically indistinguishable from random controls (McNemar's test, $p = 0.6875$, $n=149$).
- **Oracle ceiling**: Even patching *all* components simultaneously yields a $0.00\%$ fix rate.
- **Detectability ≠ Correctability**: An MLP probe flags true errors with $90.4\%$ precision. Handing these flags to the causal patch fixes **0 out of 51** true positives.
- **Collateral damage**: Patched components degrade $25.00\%$ of correct predictions vs. $0.00\%$ for random controls.

---

## Quick setup

```bash
git clone https://github.com/[YOUR_USERNAME]/[REPO_NAME].git
cd [REPO_NAME]
pip install -r requirements.txt
The model (FlameF0X/MathGPT2) is downloaded automatically via HuggingFace on first run.

Reproduce core experiments
Run the full causal screening, correction experiment, and probing baseline:

bash
# 1. Causal search (heads + MLPs with Benjamini-Hochberg correction)
python src/causal_search.py --n_samples 1411 --bh_correction

# 2. Held-out correction test (targeted vs. random-control)
python src/activation_patching.py --n_pairs 149 --position_controlled

# 3. Probing baseline (detectability)
python src/probing.py --layer 3 --classifier mlp
Citation
If you use this code, please cite:

bibtex
@article{zewdie2026where,
  title={Where Do Transformers Go Wrong? Causal Localization and Correction of Arithmetic Errors in Small Language Models},
  author={Zewdie, Yonas},
  year={2026}
}
License
MIT © Yonas Zewdie

text

---

### Why this works as a "shorter core" version:

1. **Opens with the scientific punch**: It immediately states the dissociation finding (Localization/Detection ≠ Correction) rather than burying it.
2. **Bullet-point key stats**: Uses a scannable list for all the critical numbers ($p$-values, percentages, AUC) without the overhead of a full table.
3. **Minimal setup**: Removes the virtual environment deep-dive and lengthy folder tree, keeping only the essential `git clone` and `pip install`.
4. **One-line commands**: Gives exactly 3 terminal commands to run the core experiments, showing the user what to execute without over-explaining each flag.
5. **Retains LaTeX math**: Uses `$...$` for all statistical values so they render cleanly on GitHub