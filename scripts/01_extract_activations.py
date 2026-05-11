"""Step 1: Extract anchor and axis-set activations for all models.

For each model in configs/models.yaml, forward through:
  (a) the 300-prompt HELM anchor pool (data/anchors.jsonl), and
  (b) the per-axis positive / negative prompt sets in data/axes/.

Saves one .npz per (model, collection) under outputs/activations/. The .npz
contains a single array `acts` of shape (n_prompts, n_layers + 1, d_model)
in float16. A {model}.done marker is written on success so re-runs skip.

This is the only step that needs GPUs hosting the LLMs; everything else
operates on the extracted activations.

Usage
-----
    # default: main 5-family pool, anchor pool + 10 axes (anchor + test splits),
    # plus refusal OOD splits.
    python scripts/01_extract_activations.py

    # extract only scale-sweep variants:
    python scripts/01_extract_activations.py --pool scale

    # extract a specific subset:
    python scripts/01_extract_activations.py --models llama8b qwen7b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acs.activations import extract_for_model
from acs.config import load_config
from acs.datasets import (
    DEFAULT_DATA_ROOT, AXES, OOD_VARIANTS, load_axis, load_anchors,
)


def build_prompt_collections(data_root: Path) -> dict[str, list[str]]:
    """Return a dict mapping collection name -> list of prompts.

    Collections:
      anchors            -> 300 anchor prompts
      {axis}_anchor_{pos,neg}       -> 100 + 100 per axis (direction estimation)
      {axis}_test_{pos,neg}         -> 100 + 100 per axis (evaluation)
      refusal_ood_{jbb,xstest,sorrybench}_{pos,neg}   -> 100 + 100 per OOD split
    """
    cols: dict[str, list[str]] = {}
    cols["anchors"] = load_anchors(data_root)

    for axis in AXES:
        for split in ("anchor", "test"):
            pos, neg = load_axis(axis, split=split, data_root=data_root)
            cols[f"{axis}_{split}_pos"] = pos
            cols[f"{axis}_{split}_neg"] = neg

    for axis, variants in OOD_VARIANTS.items():
        for ood in variants:
            pos, neg = load_axis(axis, ood_variant=ood, data_root=data_root)
            cols[f"{axis}_ood_{ood}_pos"] = pos
            cols[f"{axis}_ood_{ood}_neg"] = neg

    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pool", choices=["main", "scale"], default="main")
    ap.add_argument("--models", nargs="*", default=None,
                     help="subset of model names to extract (default: all in pool)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--output-root", default=None,
                     help="override config's output_root")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pool_key = "models_main" if args.pool == "main" else "models_scale"
    pool = cfg[pool_key]
    if args.models:
        pool = {k: v for k, v in pool.items() if k in args.models}

    output_root = Path(args.output_root or cfg["output_root"])
    act_dir = output_root / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    print(f"[01] extracting {len(pool)} models -> {act_dir}")
    prompts = build_prompt_collections(Path(args.data_root))
    print(f"[01] {len(prompts)} prompt collections "
          f"({sum(len(v) for v in prompts.values())} total prompts)")

    for name, model_cfg in pool.items():
        print(f"\n[01] === {name} ({model_cfg['hf_id']}) ===")
        extract_for_model(
            name, model_cfg, prompts, act_dir,
            batch_size=args.batch_size, device=args.device, pooling="last",
        )

    print(f"\n[01] done.")


if __name__ == "__main__":
    main()
