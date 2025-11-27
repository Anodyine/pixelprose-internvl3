import argparse
import json
from pathlib import Path


def load_ids_from_log(path: Path, field: str = "id"):
    assert path.exists(), f"Log file not found: {path}"
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.add(int(rec[field]))
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Check whether eval log contains exemplar IDs."
    )
    parser.add_argument(
        "--eval-log",
        required=True,
        help="Path to eval_*.jsonl file (e.g., logs/eval_subset0_PT_D_FS.jsonl).",
    )
    parser.add_argument(
        "--exemplar-log",
        required=True,
        help=(
            "Path to *_exemplars.jsonl file "
            "(e.g., logs/eval_subset0_PT_D_FS_exemplars.jsonl)."
        ),
    )
    args = parser.parse_args()

    eval_log_path = Path(args.eval_log)
    exemplar_log_path = Path(args.exemplar_log)

    eval_ids = load_ids_from_log(eval_log_path, field="id")
    exemplar_ids = load_ids_from_log(exemplar_log_path, field="id")

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
