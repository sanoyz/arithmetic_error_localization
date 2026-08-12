"""
Data utilities: problem generation, evaluation, and analog matching.
"""

import random
import numpy as np

from config import SEED, HARD_RANGES, BASE_TOLERANCE, TOLERANCE_FRAC, DISCOVERY_FRAC


def generate_problem_pool(seed=SEED, easy_n=80, hard_n=400):
    """Generate a pool of arithmetic problems."""
    rng = random.Random(seed)
    pool, seen = [], set()

    def add(op, a, b, difficulty):
        key = (op, a, b)
        if key in seen:
            return
        seen.add(key)
        if op == "add":
            answer, symbol = a + b, "+"
        elif op == "sub":
            answer, symbol = a - b, "-"
        else:
            answer, symbol = a * b, "*"
        pool.append({
            "op": op, "a": a, "b": b, "answer": str(answer),
            "prompt": f"{a} {symbol} {b} =", "difficulty": difficulty,
        })

    # Easy problems
    for _ in range(easy_n):
        add("add", rng.randint(1, 50), rng.randint(1, 50), "easy")
    for _ in range(easy_n):
        a = rng.randint(1, 50)
        add("sub", a, rng.randint(1, a), "easy")
    for _ in range(easy_n):
        add("mul", rng.randint(2, 20), rng.randint(2, 20), "easy")

    # Hard problems
    lo, hi = HARD_RANGES["add"]
    for _ in range(hard_n):
        add("add", rng.randint(lo, hi), rng.randint(lo, hi), "hard")
    lo, hi = HARD_RANGES["sub"]
    for _ in range(hard_n):
        a = rng.randint(lo, hi)
        add("sub", a, rng.randint(20, a), "hard")
    lo, hi = HARD_RANGES["mul"]
    for _ in range(hard_n):
        add("mul", rng.randint(lo, hi), rng.randint(lo, hi), "hard")

    rng.shuffle(pool)
    print(f"Generated {len(pool)} unique problems.")
    print(f"Hard-tier ranges: {HARD_RANGES}")
    return pool


def evaluate_pool(tokenizer, scorer, pool, device, verbose_n=20):
    """Evaluate model accuracy on problem pool."""
    print("\n" + "=" * 80)
    print("EVALUATING POOL (multi-token scoring)")
    print("=" * 80)

    for idx, p in enumerate(pool):
        input_ids = tokenizer.encode(p["prompt"], return_tensors="pt").to(device)
        answer_tokens = scorer.get_answer_tokens(p["answer"])
        matches = scorer.greedy_matches(input_ids, answer_tokens)

        p["input_ids"] = input_ids
        p["answer_tokens"] = answer_tokens
        p["correct"] = matches

        if idx < verbose_n:
            status = "OK " if matches else "ERR"
            print(f"  [{status}] {p['prompt']} -> expected {p['answer']} "
                  f"({len(answer_tokens)} tok)")

    corrects = [p for p in pool if p["correct"]]
    errors = [p for p in pool if not p["correct"]]
    
    print(f"\nScored {len(pool)} unique problems: {len(corrects)} correct, "
          f"{len(errors)} errors ({len(errors)/max(1,len(pool)):.1%} error rate)")
    
    for diff in ("easy", "hard"):
        sub = [p for p in pool if p["difficulty"] == diff]
        if sub:
            acc = sum(p["correct"] for p in sub) / len(sub)
            print(f"  {diff}: {acc:.1%} ({sum(p['correct'] for p in sub)}/{len(sub)})")
    
    for op in ("add", "sub", "mul"):
        sub = [p for p in pool if p["op"] == op]
        hard_sub = [p for p in sub if p["difficulty"] == "hard"]
        if sub:
            acc = sum(p["correct"] for p in sub) / len(sub)
            hard_acc = (sum(p["correct"] for p in hard_sub) / len(hard_sub)) if hard_sub else float("nan")
            n_err = sum(1 for p in sub if not p["correct"])
            print(f"  {op}: {acc:.1%} overall, {hard_acc:.1%} hard-tier, {n_err} errors")
    
    return pool, corrects, errors


def find_analog(err, correct_pool, scorer, base_tolerance=BASE_TOLERANCE, 
                tolerance_frac=TOLERANCE_FRAC, require_same_prompt_length=True):
    """
    Find the closest correct analog for an error.
    Matches by operation, magnitude, and optionally prompt length.
    """
    def magnitude(p):
        return p["a"] * p["b"] if p["op"] == "mul" else p["a"] + p["b"]

    err_mag = magnitude(err)
    tolerance = max(base_tolerance, tolerance_frac * err_mag)
    err_tokens = err["answer_tokens"]
    err_prompt_len = err["input_ids"].shape[1]

    candidates = []
    for c in correct_pool:
        if c["op"] != err["op"]:
            continue
        if require_same_prompt_length and c["input_ids"].shape[1] != err_prompt_len:
            continue
        c_mag = magnitude(c)
        if abs(c_mag - err_mag) <= tolerance:
            token_sim = 1.0 / (1.0 + abs(len(err_tokens) - len(c["answer_tokens"])))
            candidates.append((c, abs(c_mag - err_mag), token_sim))
    
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return candidates[0][0]


def build_analog_pairs_stratified(errors, corrects, scorer, seed=SEED, discovery_frac=DISCOVERY_FRAC):
    """Build discovery and held-out analog pairs stratified by operation."""
    rng = random.Random(seed)
    discovery_pairs, held_out_pairs = [], []
    
    print("\nPer-operation analog matching (prompt-length-matched):")
    for op in ("add", "sub", "mul"):
        op_errors = [e for e in errors if e["op"] == op]
        op_corrects = [c for c in corrects if c["op"] == op]
        rng.shuffle(op_errors)

        n_matched_strict = 0
        n_matched_loose = 0
        op_pairs = []
        
        for e in op_errors:
            strict = find_analog(e, op_corrects, scorer, require_same_prompt_length=True)
            loose = find_analog(e, op_corrects, scorer, require_same_prompt_length=False)
            if loose is not None:
                n_matched_loose += 1
            if strict is not None:
                n_matched_strict += 1
                op_pairs.append((e, strict))

        n_disc = max(1, int(len(op_pairs) * discovery_frac)) if op_pairs else 0
        disc, held = op_pairs[:n_disc], op_pairs[n_disc:]
        discovery_pairs.extend(disc)
        held_out_pairs.extend(held)
        
        dropped = n_matched_loose - n_matched_strict
        drop_pct = (dropped / n_matched_loose) if n_matched_loose else 0.0
        print(f"  {op}: {len(op_errors)} errors -> {n_matched_loose} matched without length "
              f"constraint, {n_matched_strict} with it ({dropped} dropped, {drop_pct:.1%}) "
              f"-> {len(disc)} discovery, {len(held)} held-out")

    rng.shuffle(discovery_pairs)
    rng.shuffle(held_out_pairs)
    return discovery_pairs, held_out_pairs