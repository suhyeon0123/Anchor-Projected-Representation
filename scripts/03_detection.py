"""Step 3: Cross-family detection (paper Section 6).

5-rotation: each of the 5 main models is held out as the unseen target once;
the remaining 4 serve as sources. For each rotation we train two probes on
source-side ACS-projected representations:

  - mCLS-LR : a single 10-way logistic-regression probe over axis labels,
              trained on positive examples only.
  - BCLS-LR : one binary probe per axis (positive vs negative), 10 in total.

Both probes are evaluated on held-out test prompts from the target model's
test split. Per-axis recall (mCLS) and AUROC (BCLS) plus the aggregate
10-way accuracy and mean AUROC are saved.

Optional flag --ood adds refusal-axis evaluation on three OOD splits
(JailbreakBench, XSTest, SORRY-Bench) using the trained refusal probe
without retraining.

Outputs
-------
    outputs/detection/per_axis_per_unseen.json
    outputs/detection/5rotation_aggregate.json
    outputs/detection/refusal_ood.json     (when --ood is set)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acs.config import load_config
from acs.datasets import OOD_VARIANTS, AXES
from acs.projector import AnchorProjector


def load_npz_layer(p: Path, layer: int) -> np.ndarray:
    arr = np.load(p)["acts"].astype(np.float32)[:, layer, :]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def project_set(act_dir: Path, model: str, layer: int, collection: str,
                 projector: AnchorProjector, device: str) -> np.ndarray:
    """Project an entire prompt set into ACS using the model's anchor projector."""
    arr = load_npz_layer(act_dir / f"{model}__{collection}.npz", layer)
    t = torch.from_numpy(arr).to(device)
    r = projector(t, treat_as_point=True)
    r = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
    return r.cpu().numpy()


def build_projector(act_dir: Path, model: str, layer: int,
                     device: str) -> AnchorProjector:
    arr = load_npz_layer(act_dir / f"{model}__anchors.npz",
                          layer)
    return AnchorProjector(torch.from_numpy(arr).to(device))


def detection_one_rotation(unseen: str, sources: list[str],
                             cfg: dict, act_dir: Path, device: str,
                             do_ood: bool) -> dict:
    layers = {m: cfg["models_main"][m]["best_layer"]
              for m in sources + [unseen]}
    projectors = {m: build_projector(act_dir, m, layers[m], device)
                   for m in sources + [unseen]}

    # ---- training data: source-side ACS for anchor split ----
    X_train_mc, y_train_mc = [], []         # positives only (10-way)
    X_train_bc = {a: ([], []) for a in AXES}  # per-axis binary
    for m in sources:
        for ai, axis in enumerate(AXES):
            for side, label in (("pos", 1), ("neg", 0)):
                col = f"{axis}_anchor_{side}"
                r = project_set(act_dir, m, layers[m], col,
                                  projectors[m], device)
                X_train_bc[axis][0].append(r)
                X_train_bc[axis][1].append(np.full(len(r), label))
                if side == "pos":
                    X_train_mc.append(r)
                    y_train_mc.append(np.full(len(r), ai))

    X_train_mc = np.concatenate(X_train_mc, axis=0)
    y_train_mc = np.concatenate(y_train_mc, axis=0)
    mc_probe = LogisticRegression(C=1.0, max_iter=2000).fit(X_train_mc, y_train_mc)

    bc_probes = {}
    for axis in AXES:
        Xs = np.concatenate(X_train_bc[axis][0], axis=0)
        ys = np.concatenate(X_train_bc[axis][1], axis=0)
        bc_probes[axis] = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, ys)

    # ---- evaluate on unseen test split ----
    per_axis = {}
    for ai, axis in enumerate(AXES):
        r_pos = project_set(act_dir, unseen, layers[unseen],
                             f"{axis}_test_pos", projectors[unseen], device)
        r_neg = project_set(act_dir, unseen, layers[unseen],
                             f"{axis}_test_neg", projectors[unseen], device)
        # mCLS recall on positives
        pred = mc_probe.predict(r_pos)
        recall = float((pred == ai).mean())
        # BCLS AUROC
        scores_pos = bc_probes[axis].decision_function(r_pos)
        scores_neg = bc_probes[axis].decision_function(r_neg)
        y_true = np.concatenate([np.ones(len(r_pos)), np.zeros(len(r_neg))])
        y_score = np.concatenate([scores_pos, scores_neg])
        auroc = float(roc_auc_score(y_true, y_score))
        per_axis[axis] = dict(mcls_recall=recall, bcls_auroc=auroc)

    # ---- aggregates ----
    overall_acc = float(np.mean([per_axis[a]["mcls_recall"] for a in AXES]))
    mean_auroc = float(np.mean([per_axis[a]["bcls_auroc"] for a in AXES]))

    out = dict(unseen=unseen, sources=sources,
                per_axis=per_axis,
                overall_mcls_acc=overall_acc,
                mean_bcls_auroc=mean_auroc)

    if do_ood and "refusal" in OOD_VARIANTS:
        ood_out = {}
        for ood in OOD_VARIANTS["refusal"]:
            r_pos = project_set(act_dir, unseen, layers[unseen],
                                 f"refusal_ood_{ood}_pos",
                                 projectors[unseen], device)
            r_neg = project_set(act_dir, unseen, layers[unseen],
                                 f"refusal_ood_{ood}_neg",
                                 projectors[unseen], device)
            pred = mc_probe.predict(r_pos)
            recall = float((pred == AXES.index("refusal")).mean())
            scores_pos = bc_probes["refusal"].decision_function(r_pos)
            scores_neg = bc_probes["refusal"].decision_function(r_neg)
            y_true = np.concatenate([np.ones(len(r_pos)), np.zeros(len(r_neg))])
            y_score = np.concatenate([scores_pos, scores_neg])
            auroc = float(roc_auc_score(y_true, y_score))
            ood_out[ood] = dict(mcls_recall=recall, bcls_auroc=auroc)
        out["ood"] = ood_out

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ood", action="store_true",
                     help="also evaluate refusal axis on OOD splits")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.output_root or cfg["output_root"])
    act_dir = out_root / "activations"
    save_dir = out_root / "detection"
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    main_models = list(cfg["models_main"].keys())
    print(f"[03] 5-rotation detection (LQMPG)")
    rotations = []
    for unseen in main_models:
        sources = [m for m in main_models if m != unseen]
        print(f"  unseen={unseen}  sources={sources}")
        r = detection_one_rotation(unseen, sources, cfg, act_dir, device, args.ood)
        rotations.append(r)
        print(f"    -> mCLS acc={r['overall_mcls_acc']:.3f}  "
              f"BCLS mean AUROC={r['mean_bcls_auroc']:.3f}")

    json.dump(rotations, open(save_dir / "per_axis_per_unseen.json", "w"), indent=2)
    agg = dict(
        lqmp_mcls=float(np.mean([r["overall_mcls_acc"] for r in rotations
                                    if r["unseen"] != "gemma9b"])),
        lqmp_bcls=float(np.mean([r["mean_bcls_auroc"] for r in rotations
                                    if r["unseen"] != "gemma9b"])),
        lqmpg_mcls=float(np.mean([r["overall_mcls_acc"] for r in rotations])),
        lqmpg_bcls=float(np.mean([r["mean_bcls_auroc"] for r in rotations])),
    )
    json.dump(agg, open(save_dir / "5rotation_aggregate.json", "w"), indent=2)
    print(f"\n[03] 5-rotation aggregate:")
    print(f"  LQMP  (4-fam)  mCLS={agg['lqmp_mcls']:.3f}  BCLS={agg['lqmp_bcls']:.3f}")
    print(f"  LQMPG (5-fam)  mCLS={agg['lqmpg_mcls']:.3f}  BCLS={agg['lqmpg_bcls']:.3f}")

    if args.ood:
        ood_table = {r["unseen"]: r.get("ood") for r in rotations}
        json.dump(ood_table, open(save_dir / "refusal_ood.json", "w"), indent=2)
        print(f"\n[03] refusal OOD aggregate (LQMPG):")
        for split in OOD_VARIANTS["refusal"]:
            mc = [ood_table[m][split]["mcls_recall"] for m in main_models]
            bc = [ood_table[m][split]["bcls_auroc"] for m in main_models]
            print(f"  {split:<12}  mCLS={np.mean(mc):.3f}  BCLS={np.mean(bc):.3f}")

    print(f"\n[03] saved -> {save_dir}/")


if __name__ == "__main__":
    main()
