# scripts/check_exemplar_leakage.py

import argparse
import json
from pathlib import Path


def load_exemplar_ids(subset_index: int):
    ex_path = Path("prompts") / f"exemplars_subset{subset_index}.jsonl"
    assert ex_path.exists(), f"Exemplar file not found: {ex_path}"

    ids = set()
    with ex_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.add(int(rec["id"]))
    return ids


def load_eval_ids(log_path: Path):
    assert log_path.exists(), f"Log file not found: {log_path}"

    ids = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.add(int(rec["id"]))
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Check whether eval log contains exemplar IDs."
    )
    parser.add_argument(
        "--subset-index",
        type=int,
        default=0,
        help="Subset index (for prompts/exemplars_subset{index}.jsonl). Default: 0",
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to eval_*.jsonl file (e.g., logs/eval_subset0_PT_D_FS.jsonl).",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)

    exemplar_ids = load_exemplar_ids(args.subset_index)
    eval_ids = load_eval_ids(log_path)

    print(f"[INFO] Exemplar IDs      : {sorted(exemplar_ids)}")
    print(f"[INFO] Num exemplar IDs  : {len(exemplar_ids)}")
    print(f"[INFO] Num evaluated IDs : {len(eval_ids)}")

    overlap = exemplar_ids & eval_ids
    if overlap:
        print(f"[WARN] Overlap found! Evaluated exemplar IDs: {sorted(overlap)}")
    else:
        print("[OK] No exemplar IDs appear in the eval log.")


if __name__ == "__main__":
    main()
