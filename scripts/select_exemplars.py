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
    2: "detailed_fewshot.txt", # PT+D_FS / FT+D_FS (few-shot base instruction)
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


def load_exemplars(subset_index: int, max_exemplars: int = 3) -> List[Dict[str, Any]]:
    """
    Load up to max_exemplars exemplar records from
    prompts/exemplars_subset{index}.jsonl.
    """
    ex_path = Path("prompts") / f"exemplars_subset{subset_index}.jsonl"
    if not ex_path.exists():
        raise FileNotFoundError(
            f"Exemplar file not found: {ex_path}. "
            f"Run `python -m scripts.select_exemplars {subset_index} {max_exemplars}` first."
        )

    exemplars: List[Dict[str, Any]] = []
    with ex_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exemplars.append(json.loads(line))
            if len(exemplars) >= max_exemplars:
                break

    if not exemplars:
        raise ValueError(f"No exemplars found in {ex_path}")
    return exemplars


def build_fewshot_prompt(
    exemplars: List[Dict[str, Any]],
    base_instruction: str,
) -> str:
    """
    Build the text that goes after <image> for few-shot prompting.

    We treat exemplars as prior image->caption pairs.
    For now we only include the captions as text, not the exemplar images themselves.
    """
    lines = []
    lines.append(
        "You are given example image–caption pairs, followed by a new image to describe."
    )
    lines.append("")
    for i, ex in enumerate(exemplars, start=1):
        cap = ex["caption"].strip().replace("\n", " ")
        lines.append(f"Example {i} caption:")
        lines.append(cap)
        lines.append("")

    lines.append(
        "Using the same level of detail and style as the example captions above,"
    )
    lines.append(base_instruction.strip())

    return "\n".join(lines).strip()


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
            "0=neutral, 1=detailed, 2=detailed+few-shot. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate (after skipping exemplars). Default: 100",
    )

    args = parser.parse_args()

    subset_dir = Path(f"data/pixelprose_subset{args.subset_index}")
    meta_path = subset_dir / "metadata.jsonl"

    assert subset_dir.exists(), f"Subset dir not found: {subset_dir}"
    assert meta_path.exists(), f"Metadata not found: {meta_path}"

    print(f"[INFO] Using subset dir: {subset_dir}")
    print(f"[INFO] Reading metadata from: {meta_path}")

    records = load_metadata(meta_path)

    # Load base instruction prompt
    prompt_text = load_prompt(args.prompting_method)

    exemplars = []
    exemplar_ids = set()
    if args.prompting_method == 2:
        # Load exemplars and build few-shot prompt text
        exemplars = load_exemplars(args.subset_index, max_exemplars=3)
        exemplar_ids = {int(ex["id"]) for ex in exemplars}
        prompt_text = build_fewshot_prompt(exemplars, prompt_text)
        print(f"[INFO] Loaded {len(exemplars)} exemplars for few-shot prompting.")
        print(f"[INFO] Exemplar IDs (will be excluded from eval): {sorted(exemplar_ids)}")

    print(f"[INFO] Loaded {len(records)} records total")
    print(f"[INFO] Prompting method: {args.prompting_method}")
    print("[INFO] Final prompt text:")
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

    num_evaluated = 0
    max_eval = args.max_eval

    with out_path.open("w", encoding="utf-8") as outf:
        for rec in records:
            rec_id = int(rec["id"])

            # Skip exemplars when using few-shot
            if rec_id in exemplar_ids:
                continue

            if num_evaluated >= max_eval:
                break

            img_rel = rec["image_file"]
            img_path = subset_dir / img_rel
            gt_caption = rec["caption"]

            pixel_values = load_image(str(img_path), max_num=12).to(
                torch.bfloat16
            ).cuda()

            # For all methods, we send a single <image> plus the prompt text
            question = f"<image>\n{prompt_text}"

            pred_caption = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
            )

            out_rec = {
                "id": rec_id,
                "image_file": img_rel,
                "gt_caption": gt_caption,
                "prompting_method": args.prompting_method,
                "prompt_text": prompt_text,
                "model_config": model_tag,
                "prediction": pred_caption,
            }
            outf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            num_evaluated += 1
            if num_evaluated % 10 == 0:
                print(f"[INFO] Evaluated {num_evaluated}/{max_eval} examples...")

    print("[INFO] Evaluation complete.")
    print(f"[INFO] Total evaluated (excluding exemplars): {num_evaluated}")


if __name__ == "__main__":
    main()
