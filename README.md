# Detailed image captioning with InternVL 3.5 and PixelProse

## Project goals

This project asks how to make an open vision-language model produce richer, more descriptive image captions. It compares two ways to adapt **InternVL 3.5 2B Instruct** to the dense captions in **PixelProse**:

1. **Prompting:** compare a neutral instruction, a detailed instruction, and a detailed instruction with example image-caption pairs supplied at inference time.
2. **LoRA fine-tuning:** train a small adapter on PixelProse image-caption pairs, then evaluate that adapted model with the same three prompts.
3. **Evaluation:** measure similarity to held-out PixelProse captions with BLEU, METEOR, CIDEr, BERTScore, and caption length; inspect outputs for repetition and visual errors; consider the memory cost of including example images.

The central question is whether prompt examples, parameter-efficient training, or their combination offers the best improvement for the available compute. The [project report](report.tex) describes the experiment and findings. In that experiment, few-shot prompting improved the pretrained model most clearly, while the limited LoRA run gave modest gains and its combination with few-shot prompting sometimes produced repetitive or off-target text. Automatic scores measure agreement with *PixelProse's captions*, not independent image correctness.

## Set up

Run commands from the repository root. The scripts use relative paths such as `data/`, `prompts/`, `logs/`, and `checkpoints/`.

You need Python, a CUDA-capable GPU with enough memory for InternVL 3.5 2B, and internet access for the Hugging Face model, PixelProse dataset, source images, and metric downloads. Evaluation calls `.cuda()` directly; LoRA training is configured for a single GPU. Few-shot evaluation processes the exemplar images and query image together, so it needs substantially more memory than single-image captioning. Model loading uses `trust_remote_code=True`; review the upstream model code if your environment requires it.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` is a snapshot of the project's Python environment, including CUDA 12.4 PyTorch builds. Use a compatible Python, GPU driver, and package index for those wheels. Large downloads and GPU jobs are expected; the repository does not include the full dataset, model weights, or trained adapter.

## Run the pipeline

Each step below says what it produces and why it matters to the comparison. The examples use **subset 2** throughout so training and evaluation refer to the same local dataset.

### 1. Check the base model and image preprocessing

```bash
python -m src.test_internvl_caption
```

This loads `data/test.jpg`, applies the dynamic image tiling used throughout the project, and prints captions from neutral and detailed prompts. It is a quick check that the model, tokenizer, image processing, and GPU work before a larger run. Replace `data/test.jpg` if you want to try another image; this script has no image-path argument.

### 2. Build a PixelProse subset

```bash
python -m scripts.build_pixelprose_subset 2 3500
```

The first argument is the subset index; the second is the target number of **successfully downloaded** images. The script loads the PixelProse train split from Hugging Face, shuffles it with seed `42 + index`, downloads images from their source URLs, and writes `data/pixelprose_subset2/images/` plus `metadata.jsonl`. Each metadata line contains a local ID, image path, caption, and source metadata. Failed or missing images are skipped, so the actual count can be lower than requested. Re-running the command rewrites that subset's metadata and images.

The first 1,000 local IDs are intended for final evaluation. Training uses IDs **1,000 and above**. For the default training split of 2,000 train and 500 validation images, request at least 3,500 successful downloads. The builder loads the full dataset before selecting images, so allow for substantial disk space and download time. Downloaded subsets are ignored by Git.

### 3. Train a LoRA adapter

```bash
python -m src.train_lora \
  --subset-index 2 \
  --train-size 2000 --val-size 500 \
  --batch-size 4 --num-epochs 1 --max-steps 50
```

This tests whether changing a small set of language-model weights improves dense captioning. It applies LoRA to InternVL's language-model attention and MLP projections, using the detailed prompt and PixelProse captions. It saves an adapter, by default, to `checkpoints/internvl3_5_2b_lora_pixelprose/subset2_r32_a64/`. Use `--output-dir PATH` to choose another location, `--resume-from PATH` to continue an adapter, and `--cuda-device N` to choose the training GPU. `--max-length` controls token truncation; `--max-steps` caps optimization steps even if epochs remain.

The script partitions IDs ≥ 1,000 in metadata order: the first `--train-size` for training and the next `--val-size` for validation. It requires both partitions to be nonempty. **The current training loop does not evaluate on the validation partition**; it only reports training loss and saves the adapter. To keep evaluation held out, choose `--max-eval` and `--num-exemplars` below so their combined count is at most 1,000, and keep training on the same subset.

### 4. Evaluate pretrained and fine-tuned captioning

```bash
python -m src.eval_captions --subset-index 2 --max-eval 500 --num-exemplars 2
python -m src.eval_captions --subset-index 2 --max-eval 500 --num-exemplars 2 --use-finetuned
```

The first command tests the pretrained model (`PT`); the second loads the saved LoRA adapter (`FT`). If training used a custom output directory, set `INTERNVL_FINETUNED_PATH` for the second command:

```bash
INTERNVL_FINETUNED_PATH=/path/to/adapter python -m src.eval_captions \
  --subset-index 2 --max-eval 500 --num-exemplars 2 --use-finetuned
```

Each run evaluates **all three** prompt styles on the same images: `N` (neutral), `D` (detailed), and `D_FS` (detailed with multimodal exemplars). The exemplar count defaults to 2; the report's final few-shot experiment describes 3 exemplars, which you can select with `--num-exemplars 3`. The script takes the first records as exemplars, excludes their IDs from evaluation, then evaluates up to `--max-eval` remaining records in metadata order. It does not itself restrict evaluation to IDs below 1,000; the count rule above maintains the intended holdout on a freshly built subset.

Outputs are JSONL files in `logs/`, for example `eval_subset2_PT_N.jsonl`, `eval_subset2_PT_D.jsonl`, `eval_subset2_PT_D_FS.jsonl`, and corresponding `FT` files. `eval_subset2_PT_exemplars.jsonl` and `eval_subset2_FT_exemplars.jsonl` record the chosen examples. These logs let the same predictions be scored and inspected without rerunning the model.

### 5. Check exemplar separation

```bash
python -m scripts.check_exemplar_leakage \
  --eval-log logs/eval_subset2_PT_D_FS.jsonl \
  --exemplar-log logs/eval_subset2_PT_exemplars.jsonl
```

This prints whether exemplar IDs appear among evaluated IDs. Use the matching `FT` files to check a fine-tuned run. The check supports the comparison by confirming that the examples shown in the few-shot prompt were excluded from the scored queries. It reports overlap but does not fail with a nonzero exit code when overlap exists.

### 6. Compute comparable metrics

```bash
python -m scripts.compute_metrics \
  --logs logs/eval_subset2_PT_N.jsonl logs/eval_subset2_PT_D.jsonl logs/eval_subset2_PT_D_FS.jsonl \
         logs/eval_subset2_FT_N.jsonl logs/eval_subset2_FT_D.jsonl logs/eval_subset2_FT_D_FS.jsonl \
  --out-csv metrics/metrics_subset2_PT_FT.csv
```

This prints a table and writes a CSV with the number of records, average reference and prediction **word counts**, BLEU, METEOR, CIDEr, and BERTScore. BLEU, METEOR, and BERTScore are on a 0–100 scale in the script; CIDEr is returned on the metric implementation's native scale. BERTScore uses `bert-base-uncased` and baseline rescaling. NLTK tokenization data and the Hugging Face CIDEr/BERTScore metric implementations may download on the first run. Compare record counts before interpreting differences; partial logs can make conditions incomparable.

## Files and experiment notes

| Path | Role in the project |
| --- | --- |
| `scripts/build_pixelprose_subset.py` | Creates reproducibly shuffled local image-caption subsets. |
| `src/test_internvl_caption.py` | Shared image tiling and a quick base-model caption check. |
| `prompts/neutral.txt`, `prompts/detailed.txt`, `prompts/detailed_fewshot.txt` | The three instructions compared at evaluation time. |
| `src/lora_config.py`, `src/train_lora.py` | LoRA configuration and single-GPU training. |
| `src/eval_captions.py` | PT/FT inference and JSONL logs for all prompt styles. |
| `scripts/check_exemplar_leakage.py`, `scripts/compute_metrics.py` | Holdout check and quantitative summaries. |
| `report.tex`, `references.bib` | Full study, methodology, results, discussion, and references. |
| `logs-flawed-ft-examples/` | Tracked example logs illustrating problematic fine-tuned outputs; these are not the complete evaluation logs. |

The current `src/eval_captions.py` uses `repetition_penalty=1.15` and `no_repeat_ngram_size=4`. The final runs described in `report.tex` were made **without** those decoding controls, so new results from the current script are not exact reproductions of the report's numbers. `prompts/detailed_fewshot_anti_repetition.txt` is an experimental prompt file and is not selected by the evaluation CLI. The report contains the historical quantitative results and qualitative examples; generated datasets, checkpoints, complete logs, and metrics CSVs are not tracked in this repository.
