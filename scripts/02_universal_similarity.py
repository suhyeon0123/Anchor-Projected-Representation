"""Step 2: Cross-family universal similarity in ACS (paper Section 5).

Loads extracted activations (from step 1), builds per-model anchor projectors,
computes native axis directions, projects them into ACS, and then:

  (1) Computes the 5x5 pairwise cosine similarity matrix between
      canonical-projected directions, averaged over axes (paper Figure 2 right).
  (2) Projects all 50 (5 models x 10 axes) ACS directions to 2D via PCA / t-SNE
      and saves a scatter plot (paper Figure 2 left).

Outputs
-------
    outputs/universal/cossim_5x5.json
    outputs/universal/cossim_per_axis.json
    outputs/universal/fig_universal.png   # combined heatmap + t-SNE
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acs.canonical import canonical_direction, native_direction
from acs.config import load_config
from acs.datasets import AXES
from acs.projector import AnchorProjector


# Axes used in the paper t-SNE display (7-axis variant D).
DISPLAY_AXES = ["refusal", "math", "scireas", "factual",
                "sycophancy", "toxicity", "sentiment"]

AXIS_COLOR = {
    "refusal":    "#cc3333",
    "toxicity":   "#dd5555",
    "math":       "#aa8833",
    "scireas":    "#cc9933",
    "factual":    "#3366cc",
    "sycophancy": "#9933cc",
    "sentiment":  "#33aa33",
    "emotion":    "#55cc55",
    "bias_gender": "#990000",
    "bias_race":  "#bb2222",
}

MODEL_MARKER = {"llama8b": "o", "qwen7b": "s", "mistral7b": "^",
                 "phi4": "P", "gemma9b": "v"}
MODEL_LABEL  = {"llama8b": "Llama", "qwen7b": "Qwen", "mistral7b": "Mistral",
                 "phi4": "Phi", "gemma9b": "Gemma"}


def load_npz_layer(p: Path, layer: int) -> np.ndarray:
    arr = np.load(p)["acts"].astype(np.float32)[:, layer, :]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_projector(act_dir: Path, model: str, layer: int,
                     device: str = "cpu") -> AnchorProjector:
    arr = load_npz_layer(act_dir / f"{model}__anchors.npz",
                          layer)
    return AnchorProjector(torch.from_numpy(arr).to(device))


def compute_canonical_axis(act_dir: Path, model: str, axis: str, layer: int,
                            projector: AnchorProjector, device: str) -> np.ndarray:
    pos = load_npz_layer(act_dir / f"{model}__{axis}_anchor_pos.npz", layer)
    neg = load_npz_layer(act_dir / f"{model}__{axis}_anchor_neg.npz", layer)
    v = native_direction(pos, neg).to(device)
    u = projector(v, treat_as_point=False)
    return (u / (u.norm() + 1e-8)).cpu().numpy()


def render_heatmap(ax, M, labels):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=10, color=("white" if abs(v) > 0.55 else "black"),
                     fontweight="bold")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--axes", nargs="*", default=DISPLAY_AXES,
                     help="axes to use for the t-SNE scatter")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.output_root or cfg["output_root"])
    act_dir = out_root / "activations"
    save_dir = out_root / "universal"
    save_dir.mkdir(parents=True, exist_ok=True)

    main_models = list(cfg["models_main"].keys())
    layers = {m: cfg["models_main"][m]["best_layer"] for m in main_models}
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[02] building projectors ({len(main_models)} models)")
    projectors = {m: build_projector(act_dir, m, layers[m], device)
                   for m in main_models}

    print(f"[02] computing canonical-projected axis directions "
          f"({len(main_models)} models x {len(AXES)} axes)")
    canon = {}
    for m in main_models:
        for axis in AXES:
            canon[(m, axis)] = compute_canonical_axis(
                act_dir, m, axis, layers[m], projectors[m], device,
            )

    M_per_axis = {}
    for axis in AXES:
        M = np.zeros((len(main_models), len(main_models)))
        for i, m1 in enumerate(main_models):
            for j, m2 in enumerate(main_models):
                M[i, j] = float(canon[(m1, axis)] @ canon[(m2, axis)])
        M_per_axis[axis] = M.tolist()

    M_mean = np.mean(np.stack([np.array(M_per_axis[a]) for a in AXES]),
                       axis=0)
    json.dump(M_per_axis, open(save_dir / "cossim_per_axis.json", "w"), indent=2)
    json.dump({"axes": AXES, "models": main_models, "M": M_mean.tolist()},
               open(save_dir / "cossim_5x5.json", "w"), indent=2)

    print(f"[02] 5x5 mean-cossim heatmap (LQMPG):")
    for i, m1 in enumerate(main_models):
        row = "  " + " ".join(f"{M_mean[i, j]:+.2f}" for j in range(len(main_models)))
        print(f"  {MODEL_LABEL[m1]:<8} {row}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    render_heatmap(axes[0], M_mean, [MODEL_LABEL[m] for m in main_models])
    axes[0].set_title("Mean cossim (10 axes)")

    X, meta = [], []
    for m in main_models:
        for axis in args.axes:
            X.append(canon[(m, axis)])
            meta.append((m, axis))
    X = np.array(X)
    perp = max(2.0, min(15.0, (len(X) - 1) / 3.0))
    Z = TSNE(n_components=2, perplexity=perp, init="pca",
              learning_rate="auto", random_state=42).fit_transform(X)
    seen = set()
    for (m, axis), z in zip(meta, Z):
        lbl = axis if axis not in seen else None
        seen.add(axis)
        axes[1].scatter(z[0], z[1],
                          c=AXIS_COLOR[axis], marker=MODEL_MARKER[m],
                          s=120, alpha=0.92,
                          edgecolor="black", linewidth=0.5, label=lbl)
    axes[1].set_xlabel("t-SNE-1")
    axes[1].set_ylabel("t-SNE-2")
    axes[1].set_title(f"t-SNE of canonical-projected directions "
                       f"({len(main_models)} models, {len(args.axes)} axes)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    title="axis", fontsize=8, frameon=False)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "fig_universal.png", dpi=180, bbox_inches="tight")
    fig.savefig(save_dir / "fig_universal.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[02] saved -> {save_dir}/{{cossim_5x5.json, cossim_per_axis.json, "
          f"fig_universal.{{png,pdf}}}}")


if __name__ == "__main__":
    main()
