# src/train_lora.py
#
#   - Data loading and fixed-size split.
#   - Dataset that loads images + text.
#   - DataLoader + collate_fn that batches variable-length text and variable
#     numbers of image patches.


import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

from src.eval_captions import load_prompt
from src.test_internvl_caption import load_image
import contextlib

from peft import get_peft_model, PeftModel

from src.lora_config import make_default_lora_config
from src.test_internvl_caption import load_image


MODEL_ID = "OpenGVLab/InternVL3_5-2B-Instruct"
MIN_TRAIN_ID = 1000

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def split_train_val_by_counts(
    records: List[Dict[str, Any]],
    min_train_id: int,
    train_size: int,
    val_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter to ids >= min_train_id, then take the first train_size for train
    and the next val_size for val. Deterministic: no shuffling for now.
    """
    usable = [r for r in records if int(r["id"]) >= min_train_id]
    n_usable = len(usable)

    if n_usable == 0:
        raise RuntimeError(
            f"No usable records with id >= {min_train_id}. "
            "Check your metadata or subset index."
        )

    if train_size + val_size > n_usable:
        print(
            f"[WARN] Requested train_size ({train_size}) + val_size ({val_size}) "
            f"= {train_size + val_size} but only {n_usable} usable records exist. "
            "Truncating to fit."
        )
        train_size = min(train_size, n_usable)
        val_size = min(val_size, max(0, n_usable - train_size))

    if train_size == 0 or val_size == 0:
        raise RuntimeError(
            f"After adjustment, train_size={train_size}, val_size={val_size}. "
            "Both must be > 0."
        )

    train_records = usable[:train_size]
    val_records = usable[train_size:train_size + val_size]

    return train_records, val_records


class PixelProseLoraDataset(Dataset):
    """
    Minimal dataset for LoRA:

    - One image per example (same load_image as eval).
    - Labels are the same as input_ids.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        subset_dir: Path,
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.records = records
        self.subset_dir = subset_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Use the same detailed prompt I already use for prompting_method=1
        self.prompt_text = load_prompt(1)  # 1 = detailed.txt
        print(f"[INFO] Loaded detailed prompt for training. First line:")
        print(f"       {self.prompt_text.splitlines()[0] if self.prompt_text else ''}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_rel = rec["image_file"]
        caption = rec["caption"]

        img_path = self.subset_dir / img_rel

        pixel_values = load_image(str(img_path), max_num=12)

        question = f"<image>\n{self.prompt_text}"
        full_text = question + "\n" + caption

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        return {
            "id": int(rec["id"]),
            "image_file": img_rel,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption,
        }


def make_collate_fn(pad_token_id: int):
    """
    Collate function that:
    - Pads text to max length in batch.
    - Concatenates all pixel patches and records num_patches_list.
    """

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Text part: pad to max length
        input_ids_list = [b["input_ids"] for b in batch]
        attention_list = [b["attention_mask"] for b in batch]
        labels_list = [b["labels"] for b in batch]
        ids_list = [b["id"] for b in batch]
        image_files = [b["image_file"] for b in batch]
        captions = [b["caption"] for b in batch]

        max_len = max(x.size(0) for x in input_ids_list)

        batch_input_ids = []
        batch_attention = []
        batch_labels = []

        for inp, attn, lab in zip(input_ids_list, attention_list, labels_list):
            pad_len = max_len - inp.size(0)

            if pad_len > 0:
                pad_ids = torch.full((pad_len,), pad_token_id, dtype=inp.dtype)
                pad_attn = torch.zeros(pad_len, dtype=attn.dtype)
                pad_lab = torch.full((pad_len,), -100, dtype=lab.dtype)  # ignore pad in loss

                inp = torch.cat([inp, pad_ids], dim=0)
                attn = torch.cat([attn, pad_attn], dim=0)
                lab = torch.cat([lab, pad_lab], dim=0)

            batch_input_ids.append(inp)
            batch_attention.append(attn)
            batch_labels.append(lab)

        batch_input_ids = torch.stack(batch_input_ids, dim=0)      # [B, T]
        batch_attention = torch.stack(batch_attention, dim=0)      # [B, T]
        batch_labels = torch.stack(batch_labels, dim=0)            # [B, T]

        # Image part: variable patches per sample
        pixel_values_list = [b["pixel_values"] for b in batch]
        num_patches_list = [pv.size(0) for pv in pixel_values_list]

        # Concatenate along patch dimension
        pixel_values_all = torch.cat(pixel_values_list, dim=0)

        return {
            "ids": ids_list,
            "image_files": image_files,
            "captions": captions,
            "pixel_values": pixel_values_all,
            "num_patches_list": num_patches_list,
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
        }

    return collate

def build_mm_inputs_for_batch(
    tokenizer,
    captions,
    num_patches_list,
    prompt_text: str,
    num_image_token: int,
    max_length: int,
    device: torch.device,
):
    """
    Build input_ids / attention_mask / labels with the correct number of
    <IMG_CONTEXT> tokens per sample, so InternVL can inject visual features.
    """
    queries = []
    for cap, num_patches in zip(captions, num_patches_list):
        # How many visual tokens this sample needs
        n_vis_tokens = num_image_token * num_patches

        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * n_vis_tokens
            + IMG_END_TOKEN
        )

        # Simple instruction format: [image tokens] + prompt + caption
        text = (
            image_tokens
            + "\n"
            + prompt_text.strip()
            + "\n"
            + cap
        )
        queries.append(text)

    # Tokenize as a batch
    enc = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone() 

    return input_ids, attention_mask, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA training setup: data loading, fixed-size split, dataset, "
            "and a single-batch DataLoader sanity check."
        )
    )
    parser.add_argument(
        "--subset-index",
        type=int,
        default=2,
        help="PixelProse subset index (uses data/pixelprose_subset{index}). Default: 2",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=2000,
        help="Number of train samples to take from id >= 1000 pool. Default: 2000",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Number of val samples to take from id >= 1000 pool. Default: 500",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for tokenizer. Default: 512",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader sanity check. Default: 4",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs over the train set. Default: 1",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Optional cap on training steps (batches) for quick runs. Default: 50",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="Which CUDA device to run on (single-GPU). Default: 0",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to an existing LoRA adapter to continue training from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the LoRA adapter. "
            "If not set, a default path based on subset index is used.",
    )

    args = parser.parse_args()

    subset_dir = Path(f"data/pixelprose_subset{args.subset_index}")
    meta_path = subset_dir / "metadata.jsonl"

    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset dir not found: {subset_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    print(f"[INFO] Using subset dir: {subset_dir}")
    print(f"[INFO] Reading metadata from: {meta_path}")

    records = load_metadata(meta_path)
    print(f"[INFO] Loaded {len(records)} records total")

    # Explicitly count how many are reserved for final eval
    reserved = [r for r in records if int(r["id"]) < MIN_TRAIN_ID]
    print(f"[INFO] Reserved for final PT/FT eval (id < {MIN_TRAIN_ID}): {len(reserved)}")

    train_records, val_records = split_train_val_by_counts(
        records,
        min_train_id=MIN_TRAIN_ID,
        train_size=args.train_size,
        val_size=args.val_size,
    )

    print(f"[INFO] Usable for LoRA (id >= {MIN_TRAIN_ID}): {len(train_records) + len(val_records)}")
    print(f"[INFO] Train records (requested {args.train_size}): {len(train_records)}")
    print(f"[INFO] Val records   (requested {args.val_size}):  {len(val_records)}")

    #  show id ranges
    if train_records:
        print(
            f"[DEBUG] Train id range: "
            f"{min(int(r['id']) for r in train_records)}"
            f" .. {max(int(r['id']) for r in train_records)}"
        )
    if val_records:
        print(
            f"[DEBUG] Val id range:   "
            f"{min(int(r['id']) for r in val_records)}"
            f" .. {max(int(r['id']) for r in val_records)}"
        )

    # Load tokenizer
    print(f"[INFO] Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    print(f"[INFO] Using pad_token_id={pad_token_id}")

    # Build training dataset
    train_ds = PixelProseLoraDataset(
        train_records,
        subset_dir=subset_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    # Save prompt + max_length for multimodal input building later
    train_prompt_text = train_ds.prompt_text
    train_max_length = args.max_length

    print(f"[INFO] Built training dataset with {len(train_ds)} samples.")

    # DataLoader + single-batch sanity check
    collate_fn = make_collate_fn(pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    print("[INFO] Fetching one batch from DataLoader for sanity check...")
    debug_batch = next(iter(train_loader))

    print("\n[INFO] Debug batch summary:")
    print(f"  ids:                 {debug_batch['ids']}")
    print(f"  num_patches_list:    {debug_batch['num_patches_list']}")
    print(f"  pixel_values shape:  {tuple(debug_batch['pixel_values'].shape)}  "
          "(sum_patches, 3, H, W)")
    print(f"  input_ids shape:     {tuple(debug_batch['input_ids'].shape)}      (B, T)")
    print(f"  attention_mask shape:{tuple(debug_batch['attention_mask'].shape)} (B, T)")
    print(f"  labels shape:        {tuple(debug_batch['labels'].shape)}         (B, T)")
    print(f"  attention_mask sums: "
          f"{[int(x) for x in debug_batch['attention_mask'].sum(dim=1)]}")

    # --------------------------------------------------------
    # Load base InternVL model and wrap language_model with LoRA
    # --------------------------------------------------------
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model_id = MODEL_ID

    print(f"[INFO] Loading base InternVL model for LoRA on {device}: {model_id}")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map=None,          # SINGLE GPU
    )

    # Build or resume LoRA adapter on the language_model (Qwen LM)
    if args.resume_from:
        print(f"[INFO] Resuming LoRA from adapter at: {args.resume_from}")
        lm = PeftModel.from_pretrained(
            model.language_model,
            args.resume_from,
            is_trainable=True,  # keep LoRA params trainable
        )
    else:
        print("[INFO] Creating new LoRA adapter for language_model...")
        lora_config = make_default_lora_config()
        lm = get_peft_model(model.language_model, lora_config)

    lm.print_trainable_parameters()
    model.language_model = lm  # plug LoRA LM back into InternVL

    model.to(device)
    model.train()

    # Tell InternVL which token is the image-context marker
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    print(f"[INFO] IMG_CONTEXT_TOKEN id: {img_context_token_id}, "
          f"num_image_token: {model.num_image_token}")

    # Optimizer over LoRA params only
    optimizer = torch.optim.AdamW(
        [p for p in model.language_model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    if device.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    global_step = 0
    print(f"[INFO] Starting training for {args.num_epochs} epoch(s), "
          f"max_steps={args.max_steps}...")
    train_start_time = time.time()

    for epoch in range(args.num_epochs):
        print(f"\n[INFO] Epoch {epoch + 1}/{args.num_epochs}")
        for step, batch in enumerate(train_loader):
            if global_step >= args.max_steps:
                print(f"[INFO] Reached max_steps={args.max_steps}, stopping training.")
                break

            model.train()

            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            num_patches_list = batch["num_patches_list"]
            captions = batch["captions"]

            input_ids, attention_mask, labels = build_mm_inputs_for_batch(
                tokenizer=tokenizer,
                captions=captions,
                num_patches_list=num_patches_list,
                prompt_text=train_prompt_text,
                num_image_token=model.num_image_token,
                max_length=train_max_length,
                device=device,
            )

            image_flags = torch.ones(
                pixel_values.size(0), 1,
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    image_flags=image_flags,
                )
                loss = outputs.loss

            loss_value = float(loss.item())
            loss.backward()

            # Optional grad norm monitor
            if (global_step % 10) == 0:
                total_norm = 0.0
                count = 0
                for p in model.language_model.parameters():
                    if p.requires_grad and p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        count += 1
                if count > 0:
                    total_norm = total_norm ** 0.5
                print(f"[STEP {global_step}] loss={loss_value:.4f}, "
                      f"grad_norm={total_norm:.4f}")

            optimizer.step()
            global_step += 1

        if global_step >= args.max_steps:
            break

    print(f"\n[INFO] Training finished at global_step={global_step}.")
    
    # -------------------------------
    # Timing summary
    # -------------------------------
    train_end_time = time.time()
    elapsed = train_end_time - train_start_time
    mins = elapsed / 60
    hrs = mins / 60
    print("[INFO] ------------------------------------------")
    print(f"[INFO] Training elapsed time: {elapsed:.2f} seconds")
    print(f"[INFO] = {mins:.2f} minutes")
    print(f"[INFO] = {hrs:.2f} hours")
    print("[INFO] ------------------------------------------")

    # --------------------------------------------------------
    # Save LoRA adapter (language_model only)
    # --------------------------------------------------------

    # Choose output directory
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        # Default path if none is provided
        out_dir = Path(
            f"checkpoints/internvl3_5_2b_lora_pixelprose/subset{args.subset_index}_r32_a64"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving LoRA adapter to: {out_dir}")
    model.language_model.save_pretrained(out_dir)
    print("[INFO] Done. You can now load this adapter in eval_captions.py via PeftModel.from_pretrained(base_model, <adapter_path>).")

if __name__ == "__main__":
    main()
