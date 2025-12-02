#!/usr/bin/env python
# scripts/compute_metrics.py
#
# Compute automatic metrics for one or more eval logs.
# Currently: BLEU only (via sacrebleu).
# Design: to add a new metric (e.g., METEOR), you:
#   1) Implement compute_meteor(preds, refs)
#   2) Register it in METRIC_FNS = {"bleu": compute_bleu, "meteor": compute_meteor, ...}
#
# Usage:
#   python -m scripts.compute_metrics \
#       --logs logs/eval_subset0_PT_N.jsonl \
#              logs/eval_subset0_PT_D.jsonl \
#              logs/eval_subset0_PT_D_FS.jsonl \
#       --out-csv metrics/metrics_subset0_PT.csv

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Callable

import sacrebleu
import nltk
nltk.data.path.insert(0, ".venv/nltk_data")
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import evaluate



# ---------------------------------------------------------------------
# Metric functions (all share the same signature)
# ---------------------------------------------------------------------

def compute_bleu(preds: List[str], refs: List[str]) -> float:
    """
    Corpus BLEU using sacrebleu.
    preds: list of model predictions (strings)
    refs:  list of reference captions (strings)
    """
    if not preds:
        return 0.0
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return float(bleu.score)

def compute_meteor(preds: List[str], refs: List[str]) -> float:
    """
    Corpus METEOR as the average of sentence-level METEOR scores.
    Returns score on 0-100 scale for consistency with BLEU.
    """
    if not preds:
        return 0.0

    scores = []
    for hyp, ref in zip(preds, refs):
        hyp = hyp or ""
        ref = ref or ""

        # Tokenize both hypothesis and reference
        hyp_tokens = word_tokenize(hyp)
        ref_tokens = word_tokenize(ref)

        # meteor_score expects token lists:
        # references: List[List[str]], hypothesis: List[str]
        scores.append(meteor_score([ref_tokens], hyp_tokens))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return float(avg_score * 100.0)

# Global handle for CIDEr metric (lazy-loaded)
_CIDER_METRIC = None

def get_cider_metric():
    global _CIDER_METRIC
    if _CIDER_METRIC is None:
        # Uses the Kamichanw/CIDEr metric on the Hub
        _CIDER_METRIC = evaluate.load("Kamichanw/CIDEr")
    return _CIDER_METRIC

def compute_cider(preds: List[str], refs: List[str]) -> float:
    """
    Corpus CIDEr using the Hugging Face evaluate implementation.

    preds: list of hypothesis captions (strings)
    refs:  list of reference captions (strings)

    We have exactly one reference per prediction, so we wrap each ref
    in a singleton list, as the metric expects List[List[str]].
    """
    if not preds:
        return 0.0

    metric = get_cider_metric()

    # Metric expects:
    #   predictions: List[str]
    #   references: List[List[str]]  (list of reference captions per prediction)
    references_wrapped = [[r or ""] for r in refs]
    predictions = [p or "" for p in preds]

    result = metric.compute(predictions=predictions, references=references_wrapped)
    # The metric returns something like {"CIDEr": value}
    score = float(result.get("CIDEr", 0.0))

    return score
# Global handle for BERTScore metric (lazy-loaded)
_BERTSCORE_METRIC = None

def get_bertscore_metric():
    global _BERTSCORE_METRIC
    if _BERTSCORE_METRIC is None:
        # This uses the 'bert-score' metric from Hugging Face evaluate
        _BERTSCORE_METRIC = evaluate.load("bertscore")
    return _BERTSCORE_METRIC

def compute_bertscore(preds: List[str], refs: List[str]) -> float:
    """
    Corpus BERTScore (F1), averaged over all examples.
    Returns score on 0-100 scale for consistency with BLEU/METEOR/CIDEr.
    """
    if not preds:
        return 0.0

    metric = get_bertscore_metric()

    # BERTScore expects:
    #   predictions: List[str]
    #   references:  List[str]
    result = metric.compute(
        predictions=[p or "" for p in preds],
        references=[r or "" for r in refs],
        lang="en",
        model_type="bert-base-uncased",  # lighter, good enough for this project
        rescale_with_baseline=True,
    )

    # result["f1"] is a list of per-example scores in [0, 1]
    f1_scores = result["f1"]
    avg_f1 = sum(f1_scores) / len(f1_scores)

    return float(avg_f1 * 100.0)

# Registry of metric name -> function
METRIC_FNS: Dict[str, Callable[[List[str], List[str]], float]] = {
    "bleu": compute_bleu,
    "meteor": compute_meteor,
    "cider": compute_cider,
    "bertscore": compute_bertscore,
}

# ---------------------------------------------------------------------
# Helpers to load logs and compute basic stats
# ---------------------------------------------------------------------

def load_records(path: Path) -> List[Dict[str, Any]]:
    assert path.exists(), f"Log file not found: {path}"
    recs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def avg_len(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(t.split()) for t in texts) / len(texts)


def compute_all_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Run all registered metrics in METRIC_FNS and return a dict
    {metric_name: value}.
    """
    results: Dict[str, float] = {}
    for name, fn in METRIC_FNS.items():
        results[name] = fn(preds, refs)
    return results


def summarize(path: Path) -> Dict[str, Any]:
    recs = load_records(path)

    if not recs:
        base = {
            "log": str(path),
            "prompting_method": "",
            "model_config": "",
            "num_records": 0,
            "avg_gt_len": 0.0,
            "avg_pred_len": 0.0,
        }
        # Fill metrics with zeros so CSV headers remain consistent
        for m in METRIC_FNS.keys():
            base[m] = 0.0
        return base

    gts = [r["gt_caption"] for r in recs]
    preds = [r["prediction"] for r in recs]

    avg_gt = avg_len(gts)
    avg_pred = avg_len(preds)
    metric_vals = compute_all_metrics(preds, gts)

    tag_pm = recs[0].get("prompting_method", "")
    tag_cfg = recs[0].get("model_config", "")

    row: Dict[str, Any] = {
        "log": str(path),
        "prompting_method": tag_pm,
        "model_config": tag_cfg,
        "num_records": len(recs),
        "avg_gt_len": avg_gt,
        "avg_pred_len": avg_pred,
    }
    row.update(metric_vals)
    return row


# ---------------------------------------------------------------------
# CSV + pretty table
# ---------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "log",
        "prompting_method",
        "model_config",
        "num_records",
        "avg_gt_len",
        "avg_pred_len",
    ]
    metric_fields = list(METRIC_FNS.keys())
    fieldnames = base_fields + metric_fields

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n[INFO] Wrote metrics CSV to: {out_csv}")


def print_table(rows: List[Dict[str, Any]]) -> None:
    """
    Pretty console output showing all registered metrics dynamically.
    """
    metric_names = list(METRIC_FNS.keys())  # e.g. ["bleu", "meteor"]

    print("\n=== Metric Summary ===\n")

    # Build dynamic header
    header_parts = [
        f"{'log':45}",
        f"{'PM':>2}",
        f"{'CFG':>3}",
        f"{'#':>5}",
        f"{'avg_gt':>8}",
        f"{'avg_pred':>9}",
    ]

    # Add each metric with fixed width
    for m in metric_names:
        header_parts.append(f"{m:>10}")

    header_line = "  ".join(header_parts)
    print(header_line)
    print("-" * len(header_line))

    # Table rows
    for r in rows:
        base_parts = [
            f"{Path(r['log']).name:45}",
            f"{str(r['prompting_method']):>2}",
            f"{str(r['model_config']):>3}",
            f"{r['num_records']:5d}",
            f"{r['avg_gt_len']:8.2f}",
            f"{r['avg_pred_len']:9.2f}",
        ]

        metric_parts = [f"{r[m]:10.2f}" for m in metric_names]
        line = "  ".join(base_parts + metric_parts)
        print(line)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics and caption length statistics for eval logs."
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="Paths to eval logs (JSONL).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to write metrics CSV (e.g., metrics/metrics_subset0_PT.csv).",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = [summarize(Path(lp)) for lp in args.logs]

    print_table(rows)

    if args.out_csv is not None:
        write_csv(rows, Path(args.out_csv))


if __name__ == "__main__":
    main()
