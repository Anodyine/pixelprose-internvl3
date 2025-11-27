# scripts/build_pixelprose_subset.py

import json
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_image(url: str, out_path: Path, timeout: float = 10.0) -> bool:
    """Download image from URL and save as JPEG. Returns True on success, False otherwise."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"[WARN] Failed to download {url} -> {out_path}: {e}")
        return False


def main(dataset_index: int, num_samples: int):
    """
    Build a PixelProse subset with *exactly* num_samples successfully downloaded images,
    if possible.

    Args:
        dataset_index: integer index used to choose output directory and shuffle seed.
        num_samples: number of successful image downloads to collect.
    """
    # Output dir: data/pixelprose_subset{index}
    out_dir = f"data/pixelprose_subset{dataset_index}"
    seed = 42 + dataset_index

    out_dir_path = Path(out_dir)
    img_dir = out_dir_path / "images"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building PixelProse subset {dataset_index} ===")
    print(f"Output dir      : {out_dir_path}")
    print(f"Requested images: {num_samples}")
    print(f"Shuffle seed    : {seed}")

    print("Loading PixelProse from Hugging Face...")
    ds = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=False)

    print("Columns:", ds.column_names)
    total_size = len(ds)
    print(f"Dataset size    : {total_size}")

    print("Shuffling...")
    ds = ds.shuffle(seed=seed)

    meta_path = out_dir_path / "metadata.jsonl"

    kept = 0
    tried = 0

    with meta_path.open("w", encoding="utf-8") as f:
        # Iterate over the entire shuffled dataset until we have num_samples
        for ex in tqdm(ds, total=total_size, desc="Scanning dataset"):
            if kept >= num_samples:
                break

            tried += 1

            url: Optional[str] = ex.get("url")
            caption: Optional[str] = ex.get("vlm_caption") or ex.get("original_caption")

            if not url or not caption:
                # Missing necessary fields; skip
                continue

            img_fname = f"{kept:06d}.jpg"  # IDs local to this subset: 0..num_samples-1
            img_path = img_dir / img_fname

            ok = download_image(url, img_path)
            if not ok:
                continue

            record = {
                "id": kept,
                "uid": ex.get("uid"),
                "url": url,
                "image_file": str(img_path.relative_to(out_dir_path)),
                "caption": caption,
                "vlm_model": ex.get("vlm_model"),
                "aesthetic_score": ex.get("aesthetic_score"),
                "watermark_class_score": ex.get("watermark_class_score"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done scanning dataset.")
    print(f"Tried examples   : {tried}")
    print(f"Kept images      : {kept}")
    print(f"Requested images : {num_samples}")
    print(f"Data dir         : {out_dir_path}")
    print(f"Metadata         : {meta_path}")

    if kept < num_samples:
        print(
            f"[WARN] Could not collect the requested {num_samples} images. "
            f"Only {kept} valid downloads were available in the entire shuffled dataset."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build a PixelProse subset with a given index and a target number of "
            "successfully downloaded images."
        )
    )
    parser.add_argument(
        "dataset_index",
        type=int,
        help="Integer index to distinguish this subset (used in directory name and shuffle seed).",
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of successfully downloaded images to collect.",
    )

    args = parser.parse_args()
    main(args.dataset_index, args.num_samples)
