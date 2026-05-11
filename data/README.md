# Data

This directory contains all prompt-level data needed to reproduce the paper's
detection and steering experiments. **No model weights or precomputed
activations are bundled here**; those are produced by
`scripts/01_extract_activations.py`.

## Anchor pool

- `anchors.jsonl` — the fixed pool of 300 prompts sampled from
  15 HELM scenarios (20 prompts each). Each line is a JSON record:

```json
{"anchor_id": 0, "text": "...", "source_scenario": "mmlu", "tags": ["factual"]}
```

## Behavioral axes

Each axis ships two splits in `axes/`:

- `{axis}_anchor.jsonl` — 100 pairs used to estimate the native direction
  v_a^{(m)} = mean(pos) - mean(neg) on each source model.
- `{axis}_test.jsonl`   — 100 disjoint pairs used to evaluate detection on
  the held-out target model.

Each line is a paired record:

```json
{
  "axis": "refusal",
  "pair_id": 0,
  "positive": "...",
  "negative": "...",
  "pos_label": "harmful",
  "neg_label": "benign",
  "source_pos": "allenai/wildjailbreak:train:vanilla_harmful",
  "source_neg": "allenai/wildjailbreak:train:vanilla_benign"
}
```

The 10 axes are:

| Axis           | Positive source                            | Negative source                            |
|---             |---                                         |---                                         |
| `refusal`      | WildJailbreak vanilla_harmful              | WildJailbreak vanilla_benign               |
| `math`         | GSM8K test                                 | MMLU-Pro no-math                           |
| `scireas`      | GPQA gpqa-main                             | MMLU-Pro non-STEM                          |
| `factual`      | MMLU test                                  | Anthropic OpinionQA                        |
| `sycophancy`   | Sharma are-you-sure (challenge)            | same item, neutral baseline                |
| `toxicity`     | CivilComments toxic > 0.5                  | AG News                                    |
| `sentiment`    | SST-2 positive                             | SST-2 negative                             |
| `emotion`      | dair-ai/emotion anger                      | dair-ai/emotion joy                        |
| `bias_gender`  | BBQ Gender ambiguous                       | BBQ Gender disambiguated                   |
| `bias_race`    | BBQ Race ambiguous                         | BBQ Race disambiguated                     |

## Refusal OOD splits

For OOD evaluation of the refusal axis we ship three additional jsonl files,
each with 100 pairs of harmful and benign prompts:

- `refusal_ood_jbb.jsonl`        — JailbreakBench (harmful / benign)
- `refusal_ood_xstest.jsonl`     — XSTest (unsafe / safe)
- `refusal_ood_sorrybench.jsonl` — SORRY-Bench (harmful / paired Alpaca instructions)

These splits are evaluated with the *same* trained refusal classifier and the
*same* canonical refusal direction as the in-distribution split; no retraining
is performed.

## Provenance

All datasets are derived from publicly available benchmarks. We provide
deterministic seed-42 subsamples to keep the per-axis splits at exactly 100
pairs per split. The original benchmark licenses and citations are listed in
the paper's bibliography.
