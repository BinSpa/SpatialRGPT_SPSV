# Evaluation script for Molmo2-8B models on SpatialRGPT-Bench
# Based on official Molmo usage: https://huggingface.co/allenai/Molmo2-8B

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig


# Default cache directory for models
DEFAULT_CACHE_DIR = "/root/autodl-fs/models"


# System prompt for SpatialRGPT-Bench evaluation
SYSTEM_PROMPT = """You are an AI assistant for spatial reasoning evaluation.
You will be shown an image and asked a question about spatial relationships between objects or regions in the image.
The Regions in question have been annotated with their outlines and a number in the image.

Please follow these rules:
1. Answer the question directly and concisely based on what you see in the image.
2. For comparison questions (bigger/smaller, left/right, etc.), clearly state which object/region has which property.
3. For measurement questions, provide your best estimate with specific units. Use appropriate measurement units such as inches, feet, meters, centimeters, etc. based on what would be most natural for the object being measured. You may use either metric (meters, centimeters) or imperial (inches, feet) units as appropriate.

Your responses will be evaluated for accuracy against ground truth answers."""


def load_model_and_processor(model_name, cache_dir):
    """Load model and processor."""
    print(f"Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()

    print(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    return model, processor


def eval_model(args):
    """Evaluate Molmo2-8B model on SpatialRGPT-Bench."""

    model, processor = load_model_and_processor(args.model_path, args.cache_dir)

    print(f"Loading annotations from: {args.annotation_file}")
    with open(args.annotation_file) as f:
        questions = json.load(f)

    print(f"Total questions: {len(questions)}")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

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

            image_path = os.path.join(args.image_folder, 'som_images', image_id + '.png')
            image = Image.open(image_path).convert("RGB")

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{question_text}"},
                ]},
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            generated_tokens = generated_ids[:, inputs["input_ids"].size(1):]
            response = processor.post_process_image_text_to_text(
                generated_tokens, skip_special_tokens=True
            )[0]

            batch_results.append({
                "question_id": image_id,
                "question": question_text,
                "pred": response.strip(),
                "gt": ground_truth,
            })

        with open(answers_file, "a", encoding="utf-8") as f:
            for result in batch_results:
                f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        tqdm.write(f"  Saved {len(batch_results)} new results (total: {min((batch_idx + 1) * batch_size, len(questions))})...")

    print("Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Molmo2-8B models on SpatialRGPT-Bench"
    )

    parser.add_argument("--model-path", type=str, default="allenai/Molmo2-8B",
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

    args = parser.parse_args()
    eval_model(args)
