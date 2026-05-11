"""Scale sweep across model sizes (paper Section 8 / Figure 5 right).

Computes canonical-projected axis directions for all `models_scale` variants
(plus the 5 main models) and plots them in a single 2-D t-SNE embedding.

Output: outputs/sensitivity/scale_tsne.{png,pdf}
        outputs/sensitivity/scale_tsne.json   # raw 2-D coordinates
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acs.canonical import native_direction
from acs.config import load_config
from acs.projector import AnchorProjector


DEFAULT_AXES = ["refusal", "math", "scireas", "factual",
                 "sycophancy", "toxicity", "sentiment"]

AXIS_COLOR = {
    "refusal":    "#cc3333", "toxicity":   "#dd5555",
    "math":       "#aa8833", "scireas":    "#cc9933",
    "factual":    "#3366cc", "sycophancy": "#9933cc",
    "sentiment":  "#33aa33",
}

# Family-level marker. Variant size is irrelevant for marker, only family.
FAM_MARKER = {"llama": "o", "qwen": "s", "mistral": "^",
               "phi": "P", "gemma": "v"}


def family_of(model_name: str) -> str:
    for fam in FAM_MARKER:
        if model_name.startswith(fam):
            return fam
    return "?"


def load_layer(p: Path, layer: int) -> np.ndarray:
    arr = np.load(p)["acts"].astype(np.float32)[:, layer, :]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def canonical_acs(act_dir: Path, model: str, axis: str, layer: int,
                   projector: AnchorProjector, device: str) -> np.ndarray:
    pos = load_layer(act_dir / f"{model}__{axis}_anchor_pos.npz", layer)
    neg = load_layer(act_dir / f"{model}__{axis}_anchor_neg.npz", layer)
    v = native_direction(pos, neg).to(device)
    u = projector(v, treat_as_point=False)
    return (u / (u.norm() + 1e-8)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--axes", nargs="*", default=DEFAULT_AXES)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.output_root or cfg["output_root"])
    act_dir = out_root / "activations"
    save_dir = out_root / "sensitivity"
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    all_models = {**cfg["models_main"], **cfg.get("models_scale", {})}

    print(f"[scale_sweep] {len(all_models)} variants x {len(args.axes)} axes")
    canon = {}
    for m, mcfg in all_models.items():
        anchor_p = act_dir / f"{m}__anchors.npz"
        if not anchor_p.exists():
            print(f"  [skip] {m}: anchor activations missing")
            continue
        anc = load_layer(anchor_p, mcfg["best_layer"])
        proj = AnchorProjector(torch.from_numpy(anc).to(device))
        for axis in args.axes:
            canon[(m, axis)] = canonical_acs(
                act_dir, m, axis, mcfg["best_layer"], proj, device,
            )
        del proj

    keys = sorted(canon.keys(), key=lambda k: (list(all_models).index(k[0]),
                                                 args.axes.index(k[1])))
    X = np.stack([canon[k] for k in keys])
    perp = max(2.0, min(15.0, (len(X) - 1) / 3.0))
    Z = TSNE(n_components=2, perplexity=perp, init="pca",
              learning_rate="auto", random_state=42).fit_transform(X)
    print(f"  perplexity={perp:.1f}  n_points={len(X)}")

    out_json = [{"model": k[0], "axis": k[1],
                  "z1": float(z[0]), "z2": float(z[1])}
                 for k, z in zip(keys, Z)]
    json.dump(out_json, open(save_dir / "scale_tsne.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for (m, axis), z in zip(keys, Z):
        fam = family_of(m)
        ax.scatter(z[0], z[1], c=AXIS_COLOR.get(axis, "#888"),
                    marker=FAM_MARKER.get(fam, "o"),
                    s=120, alpha=0.92,
                    edgecolor="black", linewidth=0.5)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(alpha=0.3)

    axis_handles = [plt.scatter([], [], c=AXIS_COLOR[a], marker="o", s=100,
                                  edgecolor="black", linewidth=0.5,
                                  label=a) for a in args.axes]
    leg_axis = ax.legend(handles=axis_handles, title="axis",
                          loc="upper left", bbox_to_anchor=(1.02, 1.0),
                          fontsize=9, frameon=False)
    leg_axis.get_title().set_fontweight("bold")
    ax.add_artist(leg_axis)
    fam_handles = [plt.scatter([], [], c="gray", marker=FAM_MARKER[f], s=80,
                                edgecolor="black", linewidth=0.5,
                                label=f) for f in FAM_MARKER]
    ax.legend(handles=fam_handles, title="family",
              loc="lower left", bbox_to_anchor=(1.02, 0.0),
              fontsize=9, frameon=False).get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(save_dir / "scale_tsne.png", dpi=180, bbox_inches="tight")
    fig.savefig(save_dir / "scale_tsne.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[scale_sweep] saved -> {save_dir}/scale_tsne.{{png,pdf,json}}")


if __name__ == "__main__":
    main()
