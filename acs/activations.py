"""Per-layer residual-stream activation extraction.

For each model and prompt, we forward through the model and capture the
hidden state at every transformer-block output. Output shape per model:
    (n_prompts, n_layers + 1, d_model)

The framework only needs the final-token residual stream at a single
selected layer L_m, but we save all layers for downstream layer-sensitivity
experiments.
"""
from __future__ import annotations

from pathlib import Path
import gc
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(hf_id: str, dtype: str = "bfloat16",
                              device: str = "cuda:0"):
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
    )
    model.eval()
    return model, tok


def _find_layers_module(model):
    """Locate the ModuleList of transformer blocks across diverse architectures.

    Search order (most specific first):
      Gemma-3 multimodal:   model.model.language_model.layers
      Llama / Qwen / Mistral / Phi / Gemma-2:   model.model.layers
      GPT-style:            model.transformer.h
      Wrapped LM:           model.language_model.{layers, model.layers}
    """
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if (hasattr(model, "language_model")
            and hasattr(model.language_model, "model")
            and hasattr(model.language_model.model, "layers")):
        return model.language_model.model.layers
    return None


def get_n_layers(model) -> int:
    layers = _find_layers_module(model)
    if layers is None:
        raise RuntimeError("Cannot determine n_layers for this model")
    return len(layers)


def get_layer_module(model, layer_idx: int):
    layers = _find_layers_module(model)
    if layers is None:
        raise RuntimeError("Cannot determine layer module for this model")
    return layers[layer_idx]


def build_chat_prompt(tokenizer, user_msg: str) -> str:
    """Render via chat template if tokenizer has one; else return the raw text."""
    try:
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return user_msg


@torch.no_grad()
def extract_activations(model, tokenizer, prompts: list[str],
                         batch_size: int = 4, max_length: int = 1024,
                         device: str = "cuda:0",
                         pooling: str = "last") -> np.ndarray:
    """Forward prompts through the model and pool the final hidden states.

    Parameters
    ----------
    prompts : list[str]
        User messages (chat templating is applied automatically when supported).
    pooling : {"last", "mean_all"}
        "last" (default) keeps only the last non-pad token activation,
        which is what the paper uses for final-token residual streams.
        "mean_all" mean-pools over all non-pad tokens.

    Returns
    -------
    np.ndarray of shape (n_prompts, n_layers + 1, d_model), dtype float16.
    """
    rendered = [build_chat_prompt(tokenizer, p) for p in prompts]
    outs = []
    for i in tqdm(range(0, len(rendered), batch_size),
                   desc="  forward", leave=False):
        batch = rendered[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                         max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                                     # tuple, [B, T, D]
        attn = enc["attention_mask"].to(hs[0].dtype)               # [B, T]

        if pooling == "last":
            last_idx = enc["attention_mask"].sum(dim=1) - 1        # [B]
            hs_stacked = torch.stack(
                [h[torch.arange(h.size(0), device=device), last_idx, :]
                 for h in hs],
                dim=1,
            )
        elif pooling == "mean_all":
            denom = attn.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = [(h * attn.unsqueeze(-1)).sum(dim=1) / denom for h in hs]
            hs_stacked = torch.stack(pooled, dim=1)
        else:
            raise ValueError(f"unknown pooling: {pooling}")

        outs.append(hs_stacked.to(torch.float32).cpu().numpy())
        del out, hs, hs_stacked

    return np.concatenate(outs, axis=0).astype(np.float16)


def extract_for_model(model_name: str, model_cfg: dict,
                       prompts: dict[str, list[str]],
                       output_dir: Path,
                       batch_size: int = 4,
                       device: str = "cuda:0",
                       pooling: str = "last"):
    """Extract activations for one model across multiple prompt collections.

    Saves one ``{model_name}__{collection}.npz`` per collection. A
    ``{model_name}.done`` marker is written on success so re-runs skip.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done = output_dir / f"{model_name}.done"
    if done.exists():
        print(f"  [skip] {model_name}: already extracted")
        return

    model, tok = load_model_and_tokenizer(model_cfg["hf_id"],
                                            model_cfg.get("dtype", "bfloat16"),
                                            device)
    n_layers = get_n_layers(model)
    print(f"  [info] {model_name}: n_layers={n_layers}  pooling={pooling}")

    for coll, texts in prompts.items():
        t0 = time.time()
        arr = extract_activations(model, tok, texts,
                                    batch_size=batch_size,
                                    device=device,
                                    pooling=pooling)
        save_path = output_dir / f"{model_name}__{coll}.npz"
        np.savez_compressed(save_path, acts=arr)
        dt = time.time() - t0
        print(f"  [saved] {save_path}  shape={arr.shape}  ({dt:.1f}s)")

    done.touch()
    del model
    torch.cuda.empty_cache()
    gc.collect()
