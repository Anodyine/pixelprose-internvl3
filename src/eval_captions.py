# src/eval_captions.py

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoModel, AutoTokenizer

# Reuse your existing image loader from test_internvl_caption.py
from test_internvl_caption import load_image


PROMPT_FILES = {
    0: "neutral.txt",          # PT+N / FT+N
    1: "detailed.txt",         # PT+D / FT+D
    2: "detailed_fewshot.txt", # PT+D_FS / FT+D_FS (few-shot template)
}


def load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    records = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_prompt(prompting_method: int, prompt_dir: Path = Path("prompts")) -> str:
    if prompting_method not in PROMPT_FILES:
        raise ValueError(f"Unknown prompting_method {prompting_method}")

    prompt_file = prompt_dir / PROMPT_FILES[prompting_method]
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    text = prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return text


def load_model(use_finetuned: bool):
    """
    Load the base or (later) LoRA-finetuned InternVL model.

    For now, both paths load the same base InternVL 3.5 2B Instruct model.
    After you train LoRA, you can swap in PEFT here.
    """
    if use_finetuned:
        # Placeholder: later you can set INTERNVL_FINETUNED_PATH env var
        model_id = os.getenv(
            "INTERNVL_FINETUNED_PATH",
            "OpenGVLab/InternVL3_5-2B-Instruct",
        )
        print(f"[INFO] use_finetuned=True. Loading model from: {model_id}")
    else:
        model_id = "OpenGVLab/InternVL3_5-2B-Instruct"
        print(f"[INFO] use_finetuned=False. Loading pretrained model: {model_id}")

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
                    description="Evaluate InternVL captions on a PixelProse subset."
    )
    parser.add_argument(
        "--subset-index",
        type=int,
        default=0,
        help="PixelProse subset index (uses data/pixelprose_subset{index}). Default: 0",
    )
    parser.add_argument(
        "--use-finetuned",
        action="store_true",
        help="Use LoRA-finetuned model (FT) instead of pretrained (PT). Default: False",
    )
    parser.add_argument(
        "--prompting-method",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=(
            "Prompting method: "
            "0=neutral, 1=detailed, 2=detailed+few-shot template. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate. Default: 100",
    )

    args = parser.parse_args()

    subset_dir = Path(f"data/pixelprose_subset{args.subset_index}")
    meta_path = subset_dir / "metadata.jsonl"

    assert subset_dir.exists(), f"Subset dir not found: {subset_dir}"
    assert meta_path.exists(), f"Metadata not found: {meta_path}"

    print(f"[INFO] Using subset dir: {subset_dir}")
    print(f"[INFO] Reading metadata from: {meta_path}")

    records = load_metadata(meta_path)
    if args.max_eval is not None:
        records = records[: args.max_eval]

    print(f"[INFO] Loaded {len(records)} records for evaluation")

    prompt_text = load_prompt(args.prompting_method)
    print(f"[INFO] Loaded prompt for method {args.prompting_method}:")
    print("------")
    print(prompt_text)
    print("------")

    model, tokenizer = load_model(args.use_finetuned)

    # Figure out label strings for filename
    model_tag = "FT" if args.use_finetuned else "PT"
    if args.prompting_method == 0:
        prompt_tag = "N"
    elif args.prompting_method == 1:
        prompt_tag = "D"
    else:
        prompt_tag = "D_FS"

    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"eval_subset{args.subset_index}_{model_tag}_{prompt_tag}.jsonl"
    )
    print(f"[INFO] Writing outputs to: {out_path}")

    generation_config = dict(max_new_tokens=256, do_sample=False)

    with out_path.open("w", encoding="utf-8") as outf:
        for idx, rec in enumerate(records):
            img_rel = rec["image_file"]
            img_path = subset_dir / img_rel
            gt_caption = rec["caption"]

            pixel_values = load_image(str(img_path), max_num=12).to(
                torch.bfloat16
            ).cuda()

            # For now, few-shot is just a different textual instruction.
            # Later we'll extend this branch to build a conversation
            # with exemplar image–caption pairs.
            question = f"<image>\n{prompt_text}"

            pred_caption = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
            )

            out_rec = {
                "id": rec["id"],
                "image_file": img_rel,
                "gt_caption": gt_caption,
                "prompting_method": args.prompting_method,
                "prompt_text": prompt_text,
                "model_config": model_tag,
                "prediction": pred_caption,
            }
            outf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"[INFO] Processed {idx + 1}/{len(records)} examples...")

    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
