"""Step 4: Behavioral steering on an unseen model (paper Section 7).

For a target unseen model m_u, we:
  (1) Compute the canonical direction c_a in ACS from the 4 source models
      (paper default: Llama, Qwen, Phi, Gemma -> Mistral).
  (2) Reconstruct v_a^{(m_u, recon)} = A_{m_u}^T * c_a in m_u's native space.
  (3) Inject the reconstructed direction at layer L_{m_u} during generation:
          h_{L_{m_u}} <- h_{L_{m_u}} + alpha * v_a^{(m_u, recon)}
      over the alpha sweep alpha in {-5, -3, 0, +3, +5}.
  (4) Score each generated continuation using the axis-specific scorer.

The script also supports three control conditions:
    canonical : the paper's main transfer direction (described above)
    native    : v_a^{(m_u)} extracted from the unseen model itself (eval-only
                reference; not part of the transfer claim)
    random    : a unit-norm Gaussian vector (sanity control)

Outputs
-------
    outputs/steering/{unseen}_{axis}.json  per-config probe scores
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acs.activations import (
    build_chat_prompt, get_layer_module, load_model_and_tokenizer,
)
from acs.canonical import canonical_direction, native_direction, reconstruct
from acs.config import load_config
from acs.datasets import load_anchors, load_axis, AXES
from acs.projector import AnchorProjector
from acs.scorers import score_refusal_llamaguard, score_sentiment


def load_npz_layer(p: Path, layer: int) -> np.ndarray:
    arr = np.load(p)["acts"].astype(np.float32)[:, layer, :]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_projector(act_dir: Path, model: str, layer: int,
                     device: str) -> AnchorProjector:
    arr = load_npz_layer(act_dir / f"{model}__anchors.npz",
                          layer)
    return AnchorProjector(torch.from_numpy(arr).to(device))


def compute_native_dir(act_dir: Path, model: str, axis: str, layer: int,
                        device: str) -> torch.Tensor:
    pos = load_npz_layer(act_dir / f"{model}__{axis}_anchor_pos.npz", layer)
    neg = load_npz_layer(act_dir / f"{model}__{axis}_anchor_neg.npz", layer)
    return native_direction(pos, neg).to(device)


SCORERS = {
    "refusal":   ("llamaguard", 80),
    "sentiment": ("sentiment",  80),
}


@torch.no_grad()
def steered_generate(model, tokenizer, prompts: list[str], layer: int,
                      direction: torch.Tensor, alpha: float,
                      max_new_tokens: int, device: str) -> list[str]:
    block = get_layer_module(model, layer)

    def hook(_module, _inp, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * direction.to(h.dtype).to(h.device)
            return (h,) + output[1:]
        return output + alpha * direction.to(output.dtype).to(output.device)

    handle = block.register_forward_hook(hook)
    outs: list[str] = []
    try:
        for p in prompts:
            rendered = build_chat_prompt(tokenizer, p)
            enc = tokenizer(rendered, return_tensors="pt").to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
            cont = tokenizer.decode(out[0, enc["input_ids"].shape[-1]:],
                                      skip_special_tokens=True)
            outs.append(cont)
    finally:
        handle.remove()
    return outs


def score_axis(axis: str, prompts: list[str], continuations: list[str],
                device: str) -> list[float]:
    if axis == "refusal":
        return score_refusal_llamaguard(prompts, continuations, device=device)
    if axis == "sentiment":
        return score_sentiment(continuations, device=device)
    raise ValueError(f"no scorer wired for axis {axis}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--axis", default="refusal",
                     choices=list(SCORERS.keys()))
    ap.add_argument("--unseen", default=None,
                     help="target model name (default: cfg.default_unseen)")
    ap.add_argument("--alphas", nargs="*", type=float,
                     default=[-5.0, -3.0, 0.0, 3.0, 5.0])
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--scorer-device", default="cuda:1")
    ap.add_argument("--gen-device", default="cuda:0")
    ap.add_argument("--ood", default=None,
                     help="refusal OOD split (jbb / xstest / sorrybench)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.output_root or cfg["output_root"])
    act_dir = out_root / "activations"
    save_dir = out_root / "steering"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.ood is not None and args.axis != "refusal":
        raise SystemExit(
            "--ood is only supported for axis=refusal (paper Section 7.3)."
        )

    unseen = args.unseen or cfg["default_unseen"]
    main_models = list(cfg["models_main"].keys())
    sources = [m for m in main_models if m != unseen]
    layers = {m: cfg["models_main"][m]["best_layer"] for m in main_models}

    gen_device = args.gen_device if torch.cuda.is_available() else "cpu"

    # ---- canonical direction in ACS ----
    print(f"[04] building canonical direction for axis '{args.axis}' "
          f"(sources={sources} -> unseen={unseen})")
    projectors = {m: build_projector(act_dir, m, layers[m], gen_device)
                   for m in sources + [unseen]}
    native_dirs = {m: compute_native_dir(act_dir, m, args.axis, layers[m],
                                            gen_device) for m in sources}
    c_a = canonical_direction(native_dirs, {m: projectors[m] for m in sources})

    v_canon = reconstruct(c_a, projectors[unseen])
    v_native = compute_native_dir(act_dir, unseen, args.axis, layers[unseen],
                                    gen_device)
    v_native = v_native / (v_native.norm() + 1e-8)
    g = torch.Generator(device=gen_device).manual_seed(args.seed)
    v_random = torch.randn(cfg["models_main"][unseen]["d_model"],
                            device=gen_device, generator=g)
    v_random = v_random / (v_random.norm() + 1e-8)

    print(f"  cos(canonical, native) = {float(v_canon @ v_native):+.3f}")

    # ---- prompts ----
    if args.ood:
        pos, _ = load_axis(args.axis, ood_variant=args.ood)
    else:
        pos, _ = load_axis(args.axis, split="test")
    prompts = pos[: args.n_prompts]

    # ---- load model on gen_device ----
    print(f"[04] loading {unseen}")
    model, tok = load_model_and_tokenizer(cfg["models_main"][unseen]["hf_id"],
                                            cfg["models_main"][unseen]["dtype"],
                                            gen_device)

    _, max_new_tokens = SCORERS[args.axis]

    results = {"axis": args.axis, "unseen": unseen, "sources": sources,
               "ood": args.ood, "n_prompts": len(prompts),
               "cos_canon_native": float(v_canon @ v_native),
               "conditions": {}}

    for cond, vec in [("canonical", v_canon), ("native", v_native),
                       ("random", v_random)]:
        results["conditions"][cond] = {}
        for alpha in args.alphas:
            print(f"  [{cond}] alpha={alpha:+}")
            conts = steered_generate(model, tok, prompts,
                                       layers[unseen], vec, alpha,
                                       max_new_tokens, gen_device)
            scores = score_axis(args.axis, prompts, conts,
                                  device=args.scorer_device)
            mean = float(np.mean(scores))
            print(f"    mean score = {mean:+.3f}")
            results["conditions"][cond][f"{alpha:+.0f}"] = dict(
                mean=mean, scores=scores,
            )

    tag = f"{unseen}_{args.axis}" + (f"_ood_{args.ood}" if args.ood else "")
    out_path = save_dir / f"{tag}.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[04] saved -> {out_path}")


if __name__ == "__main__":
    main()
