# src/eval_captions.py
#
# Evaluate InternVL 3.5 2B on a PixelProse subset under three prompting styles:
#   0 = PT/FT + Neutral prompt (N)
#   1 = PT/FT + Detailed prompt (D)
#   2 = PT/FT + Detailed + Few-shot (multimodal) (D_FS)
#
# This script:
#   - Selects `num_exemplars` exemplars from the subset metadata.
#   - Logs exemplars to logs/eval_subset{idx}_{PT/FT}_exemplars.jsonl.
#   - Excludes exemplar IDs from evaluation.
#   - Runs all three prompting styles on the SAME set of evaluation images.
#   - Writes three eval logs:
#       logs/eval_subset{idx}_{PT/FT}_N.jsonl
#       logs/eval_subset{idx}_{PT/FT}_D.jsonl
#       logs/eval_subset{idx}_{PT/FT}_D_FS.jsonl
#
# Example:
#   python -m src.eval_captions --subset-index 0 --max-eval 100 --num-exemplars 2

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from src.test_internvl_caption import load_image


PROMPT_FILES = {
    0: "neutral.txt",          # N
    1: "detailed.txt",         # D
    2: "detailed_fewshot.txt", # D_FS instruction text
}

EXEMPLAR_MAX_NUM = 12
EXEMPLAR_INPUT_SIZE = 448


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def select_exemplars(
    records: List[Dict[str, Any]],
    num_exemplars: int,
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """
    Deterministically select the first `num_exemplars` records as exemplars.
    Returns (exemplar_records, exemplar_ids).
    """
    num_ex = min(num_exemplars, len(records))
    if num_ex <= 0:
        raise RuntimeError("No records available to select exemplars from.")

    exemplars = records[:num_ex]
    exemplar_ids = {int(r["id"]) for r in exemplars}
    return exemplars, exemplar_ids


# ---------------------------------------------------------------------------
# Evaluation for one prompting method
# ---------------------------------------------------------------------------

def evaluate_prompting_method(
    prompting_method: int,
    model,
    tokenizer,
    subset_dir: Path,
    eval_records: List[Dict[str, Any]],
    prompt_text: str,
    exemplar_pixels: List[torch.Tensor],
    exemplar_captions: List[str],
    model_tag: str,
    subset_index: int,
    max_eval: int,
) -> None:
    """
    Run evaluation for a single prompting method and write a log file.
    """
    if prompting_method == 0:
        prompt_tag = "N"
    elif prompting_method == 1:
        prompt_tag = "D"
    else:
        prompt_tag = "D_FS"

    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (
        f"eval_subset{subset_index}_{model_tag}_{prompt_tag}.jsonl"
    )
    print(f"[INFO] [{prompt_tag}] Writing outputs to: {out_path}")

    generation_config = dict(max_new_tokens=256, do_sample=False)

    num_evaluated = 0

    with out_path.open("w", encoding="utf-8") as outf:
        for rec in eval_records:
            if num_evaluated >= max_eval:
                break

            rec_id = int(rec["id"])
            img_rel = rec["image_file"]
            img_path = subset_dir / img_rel
            gt_caption = rec["caption"]

            # Load query image tiles (CPU tensor)
            query_pv = load_image(str(img_path), max_num=12)

            if prompting_method in (0, 1):
                # Single-image case
                pixel_values = query_pv.to(torch.bfloat16).cuda()

                question = f"<image>\n{prompt_text}"

                pred_caption = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                )

            else:
                # Few-shot multimodal: exemplars + query image
                assert exemplar_pixels is not None and exemplar_captions is not None

                # 1) Move exemplars and query onto GPU
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
                lines = []
                for i, cap in enumerate(exemplar_captions):
                    lines.append(f"Exemplar-{i+1}: <image>")
                    lines.append(f"Caption-{i+1}: {cap}")
                    lines.append("")

                lines.append("Query: <image>")
                lines.append(prompt_text.strip())  # detailed_fewshot instruction

                question = "\n".join(lines)

                # Debug: print patch counts
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
                "prompting_method": prompting_method,
                "prompt_text": question if prompting_method == 2 else prompt_text,
                "model_config": model_tag,
                "prediction": pred_caption,
            }
            outf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            num_evaluated += 1
            if num_evaluated % 10 == 0:
                print(f"[INFO] [{prompt_tag}] Evaluated {num_evaluated}/{max_eval} examples...")

    print(f"[INFO] [{prompt_tag}] Evaluation complete. Total evaluated: {num_evaluated}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate InternVL captions on a PixelProse subset (all prompting styles)."
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
        "--max-eval",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate (excluding exemplars). Default: 100",
    )
    parser.add_argument(
        "--num-exemplars",
        type=int,
        default=2,
        help="Number of exemplars to use for few-shot prompting. Default: 2",
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

    # 1) Select exemplars from metadata
    exemplar_recs, exemplar_ids = select_exemplars(records, args.num_exemplars)
    print(f"[INFO] Selected {len(exemplar_recs)} exemplars from metadata:")
    print(f"[INFO] Exemplar IDs (excluded from eval): {sorted(exemplar_ids)}")

    # 2) Build evaluation records (excluding exemplars)
    eval_records = [r for r in records if int(r["id"]) not in exemplar_ids]
    print(f"[INFO] Eval records after excluding exemplars: {len(eval_records)}")

    # 3) Load all prompt texts
    prompt_texts = {
        m: load_prompt(m) for m in (0, 1, 2)
    }
    for m, txt in prompt_texts.items():
        print(f"[INFO] Prompt {m} text (first line): {txt.splitlines()[0] if txt else ''}")

    # 4) Load model
    model, tokenizer = load_model(args.use_finetuned)

    model_tag = "FT" if args.use_finetuned else "PT"

    # 5) Log exemplars used for this run
    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    exemplar_log_path = out_dir / (
        f"eval_subset{args.subset_index}_{model_tag}_exemplars.jsonl"
    )
    print(f"[INFO] Writing exemplar info to: {exemplar_log_path}")
    with exemplar_log_path.open("w", encoding="utf-8") as exf:
        for ex in exemplar_recs:
            exf.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 6) Prepare exemplar tensors for few-shot prompting
    exemplar_pixels: List[torch.Tensor] = []
    exemplar_captions: List[str] = []
    for ex in exemplar_recs:
        img_rel = ex["image_file"]
        img_path = subset_dir / img_rel
        cap = ex["caption"]

        pv = load_image(
            str(img_path),
            input_size=EXEMPLAR_INPUT_SIZE,
            max_num=EXEMPLAR_MAX_NUM,
        )
        exemplar_pixels.append(pv)
        exemplar_captions.append(cap)

    print(f"[INFO] Prepared {len(exemplar_pixels)} exemplar tensors for few-shot.")

    # Clip eval_records to max_eval for consistency across all methods
    eval_records = eval_records[: args.max_eval]
    print(f"[INFO] Using {len(eval_records)} eval records for all prompting styles.")

    # 7) Run all prompting methods in sequence on the SAME eval set
    for prompting_method in (0, 1, 2):
        print(f"\n[INFO] Starting evaluation for prompting_method={prompting_method}...")
        evaluate_prompting_method(
            prompting_method=prompting_method,
            model=model,
            tokenizer=tokenizer,
            subset_dir=subset_dir,
            eval_records=eval_records,
            prompt_text=prompt_texts[prompting_method],
            exemplar_pixels=exemplar_pixels,
            exemplar_captions=exemplar_captions,
            model_tag=model_tag,
            subset_index=args.subset_index,
            max_eval=len(eval_records),
        )

    print("\n[INFO] All prompting styles evaluated.")


if __name__ == "__main__":
    main()
