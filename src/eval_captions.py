# src/eval_captions.py

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoModel, AutoTokenizer

from src.test_internvl_caption import load_image


PROMPT_FILES = {
    0: "neutral.txt",          # PT+N / FT+N
    1: "detailed.txt",         # PT+D / FT+D
    2: "detailed_fewshot.txt", # PT+D_FS / FT+D_FS (few-shot template)
}
EXEMPLAR_MAX_NUM = 12
EXEMPLAR_INPUT_SIZE = 448


def load_exemplars(
    subset_index: int,
    max_exemplars: int = 2,
) -> List[Dict[str, Any]]:
    """
    Load up to `max_exemplars` exemplar records from
    prompts/exemplars_subset{index}.jsonl.

    Each record in that file is expected to look like:
      {"id": 0, "image_file": "images/000000.jpg", "caption": "..."}

    We return a list of dicts with keys: id, image_file, caption.
    """
    exemplars_path = Path("prompts") / f"exemplars_subset{subset_index}.jsonl"
    if not exemplars_path.exists():
        raise FileNotFoundError(
            f"Exemplars file not found: {exemplars_path}. "
            "Run `python -m scripts.select_exemplars {subset_index} {max_exemplars}` first."
        )

    exemplars: List[Dict[str, Any]] = []
    with exemplars_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            exemplars.append(rec)
            if len(exemplars) >= max_exemplars:
                break

    if not exemplars:
        raise RuntimeError(f"No exemplars found in {exemplars_path}")

    return exemplars


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
    After we train LoRA, we can swap in PEFT here.
    """
    if use_finetuned:
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
            "0=neutral, 1=detailed, 2=detailed+few-shot (multimodal). "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate (excluding exemplars). Default: 100",
    )

    args = parser.parse_args()

    subset_dir = Path(f"data/pixelprose_subset{args.subset_index}")
    meta_path = subset_dir / "metadata.jsonl"

    assert subset_dir.exists(), f"Subset dir not found: {subset_dir}"
    assert meta_path.exists(), f"Metadata not found: {meta_path}"

    print(f"[INFO] Using subset dir: {subset_dir}")
    print(f"[INFO] Reading metadata from: {meta_path}")

    records = load_metadata(meta_path)
    print(f"[INFO] Loaded {len(records)} records total")

    # Load base prompt text
    prompt_text = load_prompt(args.prompting_method)
    print(f"[INFO] Loaded prompt for method {args.prompting_method}:")
    print("------")
    print(prompt_text)
    print("------")

    # Few-shot exemplars (only needed for prompting_method == 2)
    exemplars = None
    exemplar_pixels = None
    exemplar_captions = None
    exemplar_ids = set()

    if args.prompting_method == 2:
        print(f"[INFO] Loading few-shot exemplars for subset {args.subset_index}...")
        exemplars = load_exemplars(subset_index=args.subset_index, max_exemplars=2)

        exemplar_pixels = []
        exemplar_captions = []
        exemplar_ids = set()

        for ex in exemplars:
            ex_id = int(ex["id"])
            img_rel = ex["image_file"]
            img_path = subset_dir / img_rel
            cap = ex["caption"]

            pv = load_image(
                str(img_path),
                input_size=EXEMPLAR_INPUT_SIZE,
                max_num=EXEMPLAR_MAX_NUM,
            )
            exemplar_pixels.append(pv)  # still on CPU
            exemplar_captions.append(cap)
            exemplar_ids.add(ex_id)

        print(f"[INFO] Loaded {len(exemplar_pixels)} exemplars for few-shot prompting.")
        print(f"[INFO] Exemplar IDs (excluded from eval): {sorted(exemplar_ids)}")

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

    # if we are using few-shot, also log the exact exemplars used for this run
    exemplar_log_path = None
    if args.prompting_method == 2 and exemplars is not None:
        exemplar_log_path = out_dir / (
            f"eval_subset{args.subset_index}_{model_tag}_{prompt_tag}_exemplars.jsonl"
        )
        print(f"[INFO] Writing exemplar info to: {exemplar_log_path}")
        with exemplar_log_path.open("w", encoding="utf-8") as exf:
            for ex in exemplars:
                exf.write(json.dumps(ex, ensure_ascii=False) + "\n")


    generation_config = dict(max_new_tokens=256, do_sample=False)

    num_evaluated = 0
    max_eval = args.max_eval

    with out_path.open("w", encoding="utf-8") as outf:
        for rec in records:
            rec_id = int(rec["id"])

            # Skip exemplars themselves when using few-shot
            if args.prompting_method == 2 and rec_id in exemplar_ids:
                continue

            if num_evaluated >= max_eval:
                break

            img_rel = rec["image_file"]
            img_path = subset_dir / img_rel
            gt_caption = rec["caption"]

            # Load query image tiles
            query_pv = load_image(str(img_path), max_num=12)

            if args.prompting_method in (0, 1):
                # PT+N or PT+D: single-image case as before
                pixel_values = query_pv.to(torch.bfloat16).cuda()

                question = f"<image>\n{prompt_text}"

                pred_caption = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                )

            else:
                # PT+D_FS: multimodal few-shot with exemplar images + captions

                assert exemplar_pixels is not None and exemplar_captions is not None

                # 1) Move exemplars and query image onto GPU
                ex_pv_gpu = [
                    pv.to(torch.bfloat16).cuda(non_blocking=True)
                    for pv in exemplar_pixels
                ]
                query_pv_gpu = query_pv.to(torch.bfloat16).cuda(non_blocking=True)

                # 2) Concatenate patches and build num_patches_list
                pixel_values_all = torch.cat(ex_pv_gpu + [query_pv_gpu], dim=0)
                num_patches_list = [
                    pv.size(0) for pv in ex_pv_gpu
                ] + [query_pv_gpu.size(0)]

                # 3) Build multimodal few-shot prompt
                ex1_cap = exemplar_captions[0]
                ex2_cap = exemplar_captions[1] if len(exemplar_captions) > 1 else None

                lines = []
                lines.append("Exemplar-1: <image>")
                lines.append(f"Caption-1: {ex1_cap}")

                if ex2_cap is not None:
                    lines.append("")
                    lines.append("Exemplar-2: <image>")
                    lines.append(f"Caption-2: {ex2_cap}")

                lines.append("")
                lines.append("Query: <image>")
                lines.append(prompt_text.strip())  # detailed_fewshot instruction

                question = "\n".join(lines)
                
                print("----- FEWSHOT INPUT DEBUG -----")
                print(f"Num exemplars: {len(ex_pv_gpu)}")

                for i, pv in enumerate(ex_pv_gpu):
                    print(f"  Exemplar-{i+1} patches: {pv.size(0)}")

                print(f"Query patches: {query_pv_gpu.size(0)}")
                print(f"num_patches_list: {num_patches_list}")
                print(f"Total patches: {pixel_values_all.size(0)}")
                print("------------------------------")

                pred_caption = model.chat(
                    tokenizer,
                    pixel_values_all,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                )

            out_rec = {
                "id": rec_id,
                "image_file": img_rel,
                "gt_caption": gt_caption,
                "prompting_method": args.prompting_method,
                "prompt_text": question if args.prompting_method == 2 else prompt_text,
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
