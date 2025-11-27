#!/usr/bin/env python
# scripts/compute_metrics.py
#
# Compute basic automatic metrics (BLEU for now) for one or more eval logs.
#
# Usage:
#   python -m scripts.compute_metrics \
#       --logs logs/eval_subset0_PT_N.jsonl \
#              logs/eval_subset0_PT_D.jsonl \
#              logs/eval_subset0_PT_D_FS.jsonl

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import sacrebleu


def load_records(path: Path) -> List[Dict[str, Any]]:
    assert path.exists(), f"Log file not found: {path}"
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def compute_bleu(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    refs = [r["gt_caption"] for r in records]
    preds = [r["prediction"] for r in records]

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score


def avg_len(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(t.split()) for t in texts) / len(texts)


def summarize(path: Path):
    recs = load_records(path)

    gts = [r["gt_caption"] for r in recs]
    preds = [r["prediction"] for r in recs]

    bleu = compute_bleu(recs)
    avg_gt = avg_len(gts)
    avg_pred = avg_len(preds)

    # Try to extract tags
    tag_pm = recs[0].get("prompting_method", "-")
    tag_cfg = recs[0].get("model_config", "-")

    return {
        "path": str(path),
        "num": len(recs),
        "bleu": bleu,
        "avg_gt_len": avg_gt,
        "avg_pred_len": avg_pred,
        "prompting_method": tag_pm,
        "model_config": tag_cfg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute BLEU and caption length statistics for eval logs."
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="Paths to eval logs (JSONL).",
    )
    args = parser.parse_args()

    rows = []
    for lp in args.logs:
        rows.append(summarize(Path(lp)))

    # Pretty print table
    print("\n=== Metric Summary (BLEU only) ===\n")
    print(
        f"{'log':45}  {'PM':>2} {'CFG':>3} {'#':>5}  {'BLEU':>7}  "
        f"{'avg_gt':>7}  {'avg_pred':>7}"
    )
    print("-" * 90)

    for r in rows:
        print(
            f"{Path(r['path']).name:45}  "
            f"{str(r['prompting_method']):>2} "
            f"{str(r['model_config']):>3} "
            f"{r['num']:5d}  "
            f"{r['bleu']:7.2f}  "
            f"{r['avg_gt_len']:7.2f}  "
            f"{r['avg_pred_len']:7.2f}"
        )

    print("\n[INFO] Done. Add METEOR/CIDEr/BERTScore later.")


if __name__ == "__main__":
    main()
