"""Sensitivity to anchor pool size k (paper Section 8 / Figure 4 left).

For each k in {5, 10, 20, 30, 40} we sample a random subset of k anchors per
HELM scenario (so total pool size = 15 * k), rebuild the projector for each
model, and re-run 5-rotation detection. The default paper configuration is
k = 20 (=> 300 anchors).

Output: outputs/sensitivity/anchor_count.json
"""
from __future__ import annotations

import argparse
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


def load_anchor_arr(p: Path, layer: int, k_per_scenario: int = 20,
                     seed: int = 42) -> np.ndarray:
    """Subsample k_per_scenario anchors per HELM scenario from the full pool.

    The default 300-anchor file is structured as 15 scenarios * 20 prompts in
    `scenario_order` order; the first k of each block of 20 is taken when
    `k_per_scenario <= 20`. For consistency with the paper we re-shuffle within
    each scenario with seed=42 before truncating.
    """
    full = np.load(p)["acts"].astype(np.float32)[:, layer, :]
    # Assume 15 scenarios of 20 in their natural order; pick first k after a
    # deterministic shuffle within each block.
    rng = np.random.default_rng(seed)
    out = []
    for s in range(15):
        block = full[s * 20:(s + 1) * 20]
        idx = rng.permutation(20)[:k_per_scenario]
        out.append(block[idx])
    return np.concatenate(out, axis=0)


def load_layer(p: Path, layer: int) -> np.ndarray:
    arr = np.load(p)["acts"].astype(np.float32)[:, layer, :]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def project(act_dir: Path, model: str, layer: int, collection: str,
             projector: AnchorProjector, device: str) -> np.ndarray:
    arr = load_layer(act_dir / f"{model}__{collection}.npz", layer)
    t = torch.from_numpy(arr).to(device)
    r = projector(t, treat_as_point=True)
    r = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
    return r.cpu().numpy()


def detection_one(unseen: str, sources: list[str], act_dir: Path,
                   layers: dict, k: int, device: str) -> tuple[float, float]:
    projectors = {}
    for m in sources + [unseen]:
        anc = load_anchor_arr(act_dir / f"{m}__anchors.npz",
                                layers[m], k_per_scenario=k)
        projectors[m] = AnchorProjector(torch.from_numpy(anc).to(device))

    X_train_mc, y_train_mc = [], []
    X_train_bc = {a: ([], []) for a in AXES}
    for m in sources:
        for ai, axis in enumerate(AXES):
            for side, label in (("pos", 1), ("neg", 0)):
                col = f"{axis}_anchor_{side}"
                r = project(act_dir, m, layers[m], col, projectors[m], device)
                X_train_bc[axis][0].append(r)
                X_train_bc[axis][1].append(np.full(len(r), label))
                if side == "pos":
                    X_train_mc.append(r)
                    y_train_mc.append(np.full(len(r), ai))
    X_train_mc = np.concatenate(X_train_mc); y_train_mc = np.concatenate(y_train_mc)
    mc_probe = LogisticRegression(C=1.0, max_iter=2000).fit(X_train_mc, y_train_mc)
    bc_probes = {a: LogisticRegression(C=1.0, max_iter=2000).fit(
        np.concatenate(X_train_bc[a][0]), np.concatenate(X_train_bc[a][1])
    ) for a in AXES}

    recalls, aurocs = [], []
    for ai, axis in enumerate(AXES):
        rp = project(act_dir, unseen, layers[unseen],
                      f"{axis}_test_pos", projectors[unseen], device)
        rn = project(act_dir, unseen, layers[unseen],
                      f"{axis}_test_neg", projectors[unseen], device)
        recalls.append(float((mc_probe.predict(rp) == ai).mean()))
        sp = bc_probes[axis].decision_function(rp)
        sn = bc_probes[axis].decision_function(rn)
        y = np.concatenate([np.ones(len(rp)), np.zeros(len(rn))])
        s = np.concatenate([sp, sn])
        aurocs.append(float(roc_auc_score(y, s)))
    return float(np.mean(recalls)), float(np.mean(aurocs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ks", nargs="*", type=int, default=[5, 10, 20, 30, 40])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.output_root or cfg["output_root"])
    act_dir = out_root / "activations"
    save_dir = out_root / "sensitivity"
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    main_models = list(cfg["models_main"].keys())
    layers = {m: cfg["models_main"][m]["best_layer"] for m in main_models}

    results = {}
    for k in args.ks:
        print(f"\n[anchor_count] k={k}  (N = {15 * k})")
        rotations = []
        for unseen in main_models:
            sources = [m for m in main_models if m != unseen]
            mcls, bcls = detection_one(unseen, sources, act_dir, layers, k, device)
            rotations.append(dict(unseen=unseen, mcls=mcls, bcls=bcls))
            print(f"  {unseen:<10} mCLS={mcls:.3f}  BCLS={bcls:.3f}")
        results[str(k)] = dict(
            rotations=rotations,
            mcls_mean=float(np.mean([r["mcls"] for r in rotations])),
            bcls_mean=float(np.mean([r["bcls"] for r in rotations])),
        )

    json.dump(results, open(save_dir / "anchor_count.json", "w"), indent=2)
    print(f"\n[anchor_count] saved -> {save_dir/'anchor_count.json'}")


if __name__ == "__main__":
    main()
