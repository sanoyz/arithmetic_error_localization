"""
Model utilities: scoring and hooked model for causal interventions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import DEVICE, N_LAYERS, N_HEADS, HEAD_DIM


class MultiTokenScorer:
    """Handles scoring of multi-token answers."""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def get_answer_tokens(self, answer_str):
        """Get token IDs for an answer string."""
        return self.tokenizer.encode(f" {answer_str}", add_special_tokens=False)

    @torch.no_grad()
    def greedy_matches(self, input_ids, answer_tokens):
        """Check if greedy generation matches expected answer."""
        n = len(answer_tokens)
        generated = self.model.generate(
            input_ids, max_new_tokens=n, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_tokens = generated[0, input_ids.shape[1]:].tolist()
        return gen_tokens == answer_tokens

    @torch.no_grad()
    def sequence_probability(self, input_ids, answer_tokens):
        """Compute teacher-forced sequence probability."""
        full_input = torch.cat(
            [input_ids[0], torch.tensor(answer_tokens, device=input_ids.device)]
        ).unsqueeze(0)
        logits = self.model(full_input).logits[0, :-1, :]
        probs = F.softmax(logits, dim=-1)
        token_probs = []
        for i, tok_id in enumerate(answer_tokens):
            pos = input_ids.shape[1] - 1 + i
            token_probs.append(probs[pos, tok_id].item() if pos < probs.shape[0] else 0.0)
        return float(np.mean(token_probs)) if token_probs else 0.0


class MultiPositionHookedModel:
    """
    Hooked model for causal interventions at specific components and positions.
    Supports both attention heads and MLP modules.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.n_layer
        self.n_heads = model.config.n_head
        self.head_dim = model.config.n_embd // model.config.n_head

    def all_components(self):
        """Return all (layer, component) pairs: 72 heads + 6 MLPs."""
        heads = [(l, h) for l in range(self.n_layers) for h in range(self.n_heads)]
        mlps = [(l, "mlp") for l in range(self.n_layers)]
        return heads + mlps

    def _make_attn_hook(self, head_idx, position, mode, cache, key):
        """Create attention head hook."""
        n_heads, head_dim = self.n_heads, self.head_dim

        def hook(module, inputs):
            (hidden,) = inputs
            hidden = hidden.clone()
            b, t, d = hidden.shape
            pos = min(position, t - 1) if t > 0 else 0
            hv = hidden.view(b, t, n_heads, head_dim)
            if mode == "ablate":
                hv[:, pos, head_idx, :] = 0.0
            elif mode == "cache":
                cache[key] = hv[:, pos, head_idx, :].detach().clone()
            elif mode == "patch":
                hv[:, pos, head_idx, :] = cache[key]
            return (hv.view(b, t, d),)
        return hook

    def _make_mlp_hook(self, position, mode, cache, key):
        """Create MLP hook."""
        def hook(module, inputs, output):
            out = output.clone()
            b, t, d = out.shape
            pos = min(position, t - 1) if t > 0 else 0
            if mode == "ablate":
                out[:, pos, :] = 0.0
            elif mode == "cache":
                cache[key] = out[:, pos, :].detach().clone()
            elif mode == "patch":
                out[:, pos, :] = cache[key]
            return out
        return hook

    def _register(self, layer, comp, position, mode, cache=None):
        """Register a hook on a specific component."""
        key = (layer, comp, position)
        if comp == "mlp":
            module = self.model.transformer.h[layer].mlp
            return module.register_forward_hook(
                self._make_mlp_hook(position, mode, cache, key)
            )
        else:
            module = self.model.transformer.h[layer].attn.c_proj
            return module.register_forward_pre_hook(
                self._make_attn_hook(comp, position, mode, cache, key)
            )

    @torch.no_grad()
    def ablated_prob(self, input_ids, answer_tokens, scorer, layer, comp, position):
        """Get probability with a component ablated (zeroed out)."""
        h = self._register(layer, comp, position, "ablate")
        try:
            return scorer.sequence_probability(input_ids, answer_tokens)
        finally:
            h.remove()

    @torch.no_grad()
    def cache_components(self, input_ids, components, positions):
        """Cache activations for specified components and positions."""
        cache = {}
        handles = [self._register(l, c, p, "cache", cache)
                   for (l, c), p in zip(components, positions)]
        try:
            self.model(input_ids)
        finally:
            for hd in handles:
                hd.remove()
        return cache

    @torch.no_grad()
    def patched_prob(self, input_ids, answer_tokens, scorer, components, positions, cache):
        """Get probability with cached activations patched in."""
        handles = [self._register(l, c, p, "patch", cache)
                   for (l, c), p in zip(components, positions)]
        try:
            return scorer.sequence_probability(input_ids, answer_tokens)
        finally:
            for hd in handles:
                hd.remove()

    @torch.no_grad()
    def patched_matches(self, input_ids, answer_tokens, components, positions, cache):
        """Check if patched components lead to correct answer."""
        handles = [self._register(l, c, p, "patch", cache)
                   for (l, c), p in zip(components, positions)]
        try:
            generated = self.model.generate(
                input_ids, max_new_tokens=len(answer_tokens), do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id, use_cache=False,
            )
            gen_tokens = generated[0, input_ids.shape[1]:].tolist()
            return gen_tokens == answer_tokens
        finally:
            for hd in handles:
                hd.remove()