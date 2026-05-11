"""Sensitivity to source-pool size N (paper Section 8 / Figure 4 right).

For the default unseen target (Mistral-7B), we run detection with every subset
of size N in {1, 2, 3, 4} of the LQPG candidate pool (Llama, Qwen, Phi, Gemma).
Each combination's mCLS / BCLS is reported, along with the mean and std over
combinations at each N.

Output: outputs/sensitivity/source_count.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acs.config import load_config
from acs.datasets import AXES
from acs.projector import AnchorProjector


def load_layer(p: Path, layer: int) -> np.ndarray:
    arr = np.load(p)["acts"].astype(np.float32)[:, layer, :]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_projector(act_dir: Path, model: str, layer: int,
                     device: str) -> AnchorProjector:
    arr = load_layer(act_dir / f"{model}__anchors.npz", layer)
    return AnchorProjector(torch.from_numpy(arr).to(device))


def project(act_dir: Path, model: str, layer: int, collection: str,
             projector: AnchorProjector, device: str) -> np.ndarray:
    arr = load_layer(act_dir / f"{model}__{collection}.npz", layer)
    t = torch.from_numpy(arr).to(device)
    r = projector(t, treat_as_point=True)
    r = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
    return r.cpu().numpy()


def detect_with_sources(unseen: str, sources: list[str],
                          act_dir: Path, layers: dict, device: str
                          ) -> tuple[float, float]:
    projectors = {m: build_projector(act_dir, m, layers[m], device)
                   for m in sources + [unseen]}
    X_mc, y_mc = [], []
    X_bc = {a: ([], []) for a in AXES}
    for m in sources:
        for ai, axis in enumerate(AXES):
            for side, label in (("pos", 1), ("neg", 0)):
                col = f"{axis}_anchor_{side}"
                r = project(act_dir, m, layers[m], col, projectors[m], device)
                X_bc[axis][0].append(r)
                X_bc[axis][1].append(np.full(len(r), label))
                if side == "pos":
                    X_mc.append(r); y_mc.append(np.full(len(r), ai))
    X_mc = np.concatenate(X_mc); y_mc = np.concatenate(y_mc)
    mc = LogisticRegression(C=1.0, max_iter=2000).fit(X_mc, y_mc)
    bc = {a: LogisticRegression(C=1.0, max_iter=2000).fit(
        np.concatenate(X_bc[a][0]), np.concatenate(X_bc[a][1])
    ) for a in AXES}

    recalls, aurocs = [], []
    for ai, axis in enumerate(AXES):
        rp = project(act_dir, unseen, layers[unseen],
                      f"{axis}_test_pos", projectors[unseen], device)
        rn = project(act_dir, unseen, layers[unseen],
                      f"{axis}_test_neg", projectors[unseen], device)
        recalls.append(float((mc.predict(rp) == ai).mean()))
        sp = bc[axis].decision_function(rp)
        sn = bc[axis].decision_function(rn)
        y = np.concatenate([np.ones(len(rp)), np.zeros(len(rn))])
        s = np.concatenate([sp, sn])
        aurocs.append(float(roc_auc_score(y, s)))
    return float(np.mean(recalls)), float(np.mean(aurocs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--unseen", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.output_root or cfg["output_root"])
    act_dir = out_root / "activations"
    save_dir = out_root / "sensitivity"
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    unseen = args.unseen or cfg["default_unseen"]
    main_models = list(cfg["models_main"].keys())
    candidates = [m for m in main_models if m != unseen]
    layers = {m: cfg["models_main"][m]["best_layer"] for m in main_models}

    print(f"[source_count] unseen={unseen} candidates={candidates}")
    results = {}
    for N in range(1, len(candidates) + 1):
        combos = list(itertools.combinations(candidates, N))
        rows = []
        for combo in combos:
            mcls, bcls = detect_with_sources(unseen, list(combo),
                                                act_dir, layers, device)
            print(f"  N={N}  sources={'+'.join(combo)}  "
                  f"mCLS={mcls:.3f}  BCLS={bcls:.3f}")
            rows.append(dict(sources=list(combo), mcls=mcls, bcls=bcls))
        mcls_arr = np.array([r["mcls"] for r in rows])
        bcls_arr = np.array([r["bcls"] for r in rows])
        results[str(N)] = dict(
            combinations=rows,
            mcls_mean=float(mcls_arr.mean()),
            mcls_std=float(mcls_arr.std()),
            bcls_mean=float(bcls_arr.mean()),
            bcls_std=float(bcls_arr.std()),
        )
        print(f"  -> N={N} mean mCLS={mcls_arr.mean():.3f}±{mcls_arr.std():.3f}  "
              f"BCLS={bcls_arr.mean():.3f}±{bcls_arr.std():.3f}")

    json.dump(results, open(save_dir / "source_count.json", "w"), indent=2)
    print(f"\n[source_count] saved -> {save_dir/'source_count.json'}")


if __name__ == "__main__":
    main()
