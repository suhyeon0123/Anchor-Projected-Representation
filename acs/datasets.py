"""Axis dataset loader.

Each axis ships two splits (the OOD set is optional and only refusal has it):
  data/axes/{axis}_anchor.jsonl       # 100 pairs for direction estimation
  data/axes/{axis}_test.jsonl         # 100 disjoint pairs for evaluation
  data/axes/refusal_ood_{name}.jsonl  # 100 OOD pairs (refusal only)

Each jsonl line is a JSON object with at least:
  {
    "pair_id": int,
    "positive": str,     # axis-positive prompt
    "negative": str,     # axis-negative prompt
    ...
  }

Anchor pool format (300 prompts):
  data/anchors.jsonl with fields {"anchor_id": int, "text": str, ...}.
"""
from __future__ import annotations

from pathlib import Path
import json

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

AXES = [
    "refusal", "math", "scireas", "factual", "sycophancy",
    "toxicity", "sentiment", "emotion", "bias_gender", "bias_race",
]

OOD_VARIANTS = {
    "refusal": ["jbb", "xstest", "sorrybench"],
}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_anchors(data_root: Path | str = DEFAULT_DATA_ROOT) -> list[str]:
    """Load the fixed pool of 300 HELM anchor prompts."""
    rows = _read_jsonl(Path(data_root) / "anchors.jsonl")
    return [r["text"] for r in rows]


def load_axis(axis: str, split: str = "anchor",
               ood_variant: str | None = None,
               data_root: Path | str = DEFAULT_DATA_ROOT
               ) -> tuple[list[str], list[str]]:
    """Load positive/negative prompts for an axis.

    Parameters
    ----------
    axis : str
        One of AXES.
    split : {"anchor", "test"}
        "anchor" (default) is used for direction estimation; "test" for evaluation.
    ood_variant : optional[str]
        For OOD evaluation (refusal only). One of OOD_VARIANTS[axis].
        When provided, `split` is ignored.

    Returns
    -------
    pos_texts, neg_texts : list[str]
        Lists of equal length; pos[i] and neg[i] are an aligned positive/negative pair.
    """
    data_root = Path(data_root)
    if ood_variant is not None:
        assert axis in OOD_VARIANTS and ood_variant in OOD_VARIANTS[axis], (
            f"OOD variant {ood_variant!r} not available for axis {axis!r}"
        )
        rows = _read_jsonl(data_root / "axes" / f"{axis}_ood_{ood_variant}.jsonl")
    else:
        assert split in ("anchor", "test"), f"split must be anchor or test, got {split!r}"
        rows = _read_jsonl(data_root / "axes" / f"{axis}_{split}.jsonl")

    pos = [r["positive"] for r in rows]
    neg = [r["negative"] for r in rows]
    return pos, neg


def load_all_axis_prompts(split: str = "anchor",
                            data_root: Path | str = DEFAULT_DATA_ROOT
                            ) -> dict[str, list[str]]:
    """Return a {collection_name: [prompts]} mapping for one split.

    Convenience helper for activation extraction. Keys follow the pattern
    ``{axis}_{split}_{pos|neg}`` matching the npz file naming used downstream.
    """
    out = {}
    for axis in AXES:
        pos, neg = load_axis(axis, split=split, data_root=data_root)
        out[f"{axis}_{split}_pos"] = pos
        out[f"{axis}_{split}_neg"] = neg
    return out
