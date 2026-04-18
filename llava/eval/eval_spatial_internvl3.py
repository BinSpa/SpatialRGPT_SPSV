# Evaluation script for InternVL3-8B-Instruct models on SpatialRGPT-Bench
# Based on official InternVL usage: https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

import torchvision.transforms as T

# Default cache directory for models
DEFAULT_CACHE_DIR = "/root/autodl-fs/models"

IMAGE_SIZE = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# System prompt for SpatialRGPT-Bench evaluation
SYSTEM_PROMPT = """You are an AI assistant for spatial reasoning evaluation.
You will be shown an image and asked a question about spatial relationships between objects or regions in the image.
The Regions in question have been annotated with their outlines and a number in the image.

Please follow these rules:
1. Answer the question directly and concisely based on what you see in the image.
2. For comparison questions (bigger/smaller, left/right, etc.), clearly state which object/region has which property.
3. For measurement questions, provide your best estimate with specific units. Use appropriate measurement units such as inches, feet, meters, centimeters, etc. based on what would be most natural for the object being measured. You may use either metric (meters, centimeters) or imperial (inches, feet) units as appropriate.

Your responses will be evaluated for accuracy against ground truth answers."""


def build_transform(input_size=IMAGE_SIZE):
    """Build image transform for InternVL model."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def load_model_and_tokenizer(model_name, cache_dir):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    return model, tokenizer


def eval_model(args):
    """Evaluate InternVL3-8B model on SpatialRGPT-Bench."""

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.cache_dir)

    # Set system message on the model
    model.system_message = SYSTEM_PROMPT

    transform = build_transform(IMAGE_SIZE)

    print(f"Loading annotations from: {args.annotation_file}")
    with open(args.annotation_file) as f:
        questions = json.load(f)

    print(f"Total questions: {len(questions)}")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Generation config — InternVL's chat() expects a dict
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else 1.0,
        "top_p": args.top_p if args.top_p else 1.0,
    }

    batch_size = args.batch_size
    total_batches = (len(questions) + batch_size - 1) // batch_size

    print(f"Batch size: {batch_size}, Total batches: {total_batches}")
    print(f"Saving results incrementally to: {answers_file}")

    with open(answers_file, "w", encoding="utf-8") as f:
        pass  # Create/clear the file first

    for batch_idx in tqdm(range(total_batches), desc="Evaluating batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]

        batch_results = []

        for line in batch_questions:
            image_id = line["id"]
            question_text = line["text_q"]
            conversations = line["conversations"]
            ground_truth = conversations[1]["value"]

            # Load image from som_images folder
            image_path = os.path.join(args.image_folder, 'som_images', image_id + '.png')
            image = Image.open(image_path).convert("RGB")
            pixel_values = transform(image).unsqueeze(0).to(model.device).to(model.dtype)

            # Generate response using the model's chat API
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question_text,
                generation_config=generation_config,
            )

            batch_results.append({
                "question_id": image_id,
                "question": question_text,
                "pred": response.strip(),
                "gt": ground_truth,
            })

        # Save batch results
        with open(answers_file, "a", encoding="utf-8") as f:
            for result in batch_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        tqdm.write(f"  Saved {len(batch_results)} new results (total: {min((batch_idx + 1) * batch_size, len(questions))})...")

    print("Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate InternVL3-8B-Instruct models on SpatialRGPT-Bench"
    )

    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL3-8B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help=f"Model cache directory (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to benchmark root (som_images/ is appended)")
    parser.add_argument("--annotation-file", type=str, required=True,
                        help="Path to SpatialRGPT-Bench annotation JSON")
    parser.add_argument("--answers-file", type=str, required=True,
                        help="Path to output results JSONL file")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for evaluation (default: 8)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0, greedy)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p sampling parameter (default: None)")

    args = parser.parse_args()
    eval_model(args)
