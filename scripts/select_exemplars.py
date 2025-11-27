# scripts/select_exemplars.py

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    records = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main(subset_index: int, num_exemplars: int):
    subset_dir = Path(f"data/pixelprose_subset{subset_index}")
    meta_path = subset_dir / "metadata.jsonl"
    assert subset_dir.exists(), f"Subset dir not found: {subset_dir}"
    assert meta_path.exists(), f"Metadata not found: {meta_path}"

    print(f"[INFO] Loading metadata from: {meta_path}")
    records = load_metadata(meta_path)

    if num_exemplars > len(records):
        raise ValueError(
            f"Requested {num_exemplars} exemplars, but subset only has {len(records)} records"
        )

    # For now: just take the first num_exemplars.
    # Later we can change this to random or filtered selection.
    exemplars = records[:num_exemplars]

    prompts_dir = Path("prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_path = prompts_dir / f"exemplars_subset{subset_index}.jsonl"

    print(f"[INFO] Writing {len(exemplars)} exemplars to: {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for ex in exemplars:
            # Keep only what we really need
            out_rec = {
                "id": ex["id"],
                "image_file": ex["image_file"],
                "caption": ex["caption"],
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select a few exemplar image–caption pairs from a PixelProse subset."
    )
    parser.add_argument(
        "subset_index",
        type=int,
        help="Index of the subset (uses data/pixelprose_subset{index}).",
    )
    parser.add_argument(
        "num_exemplars",
        type=int,
        help="Number of exemplars to select.",
    )
    args = parser.parse_args()
    main(args.subset_index, args.num_exemplars)
