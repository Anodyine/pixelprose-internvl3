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


def main(
    num_samples: int = 2_000,
    out_dir: str = "data/pixelprose_subset",
    seed: int = 42,
):
    out_dir_path = Path(out_dir)
    img_dir = out_dir_path / "images"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PixelProse from Hugging Face...")
    ds = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=False)

    print("Columns:", ds.column_names)
    # Expect something like: ['uid', 'url', 'key', 'status', 'original_caption', 'vlm_model', 'vlm_caption', ...]

    print(f"Dataset size: {len(ds)}")
    print("Shuffling...")
    ds = ds.shuffle(seed=seed)

    num_samples = min(num_samples, len(ds))
    print(f"Selecting first {num_samples} examples...")
    ds_small = ds.select(range(num_samples))

    meta_path = out_dir_path / "metadata.jsonl"

    kept = 0
    with meta_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds_small, desc="Downloading subset")):
            url: Optional[str] = ex.get("url")
            caption: Optional[str] = ex.get("vlm_caption") or ex.get("original_caption")

            if not url or not caption:
                print(f"[WARN] Skipping sample {i} due to missing url or caption")
                continue

            img_fname = f"{kept:06d}.jpg"
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

    print(f"Done. Requested {num_samples} examples, kept {kept}.")
    print(f"Data dir: {out_dir_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
