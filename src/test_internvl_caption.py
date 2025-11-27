# src/test_internvl_caption.py

import math
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_file: str, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def main():
    # Put any local JPG/PNG here
    img_path = Path("data/test.jpg")
    assert img_path.exists(), f"Place a test image at {img_path}"

    model_id = "OpenGVLab/InternVL3_5-2B-Instruct"

    print("Loading model on GPUs...")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # keep flash-attn off for first test
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    pixel_values = load_image(str(img_path), max_num=12).to(torch.bfloat16).cuda()

    generation_config = dict(max_new_tokens=256, do_sample=False)

    # Neutral prompt (baseline)
    question_neutral = "<image>\nProvide a factual caption for this image."
    response_neutral = model.chat(
        tokenizer, pixel_values, question_neutral, generation_config
    )
    print("\n[Neutral caption]")
    print(response_neutral)

    # Detailed prompt (what you care about)
    question_detailed = (
        "<image>\nProvide a detailed, comprehensive caption describing all key "
        "objects, attributes, actions, and context."
    )
    response_detailed = model.chat(
        tokenizer, pixel_values, question_detailed, generation_config
    )
    print("\n[Detailed caption]")
    print(response_detailed)


if __name__ == "__main__":
    main()
