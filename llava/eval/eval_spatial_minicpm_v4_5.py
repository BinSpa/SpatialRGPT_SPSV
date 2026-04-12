# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Evaluation script for MiniCPM-V-4_5 models on SpatialRGPT-Bench
# Based on official MiniCPM-V usage: https://huggingface.co/openbmb/MiniCPM-V-4_5

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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


def load_model_and_tokenizer(model_name, cache_dir, local_only=False):
    """Load model and tokenizer, downloading if necessary."""
    print(f"Loading model: {model_name}")

    load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16,
        "cache_dir": cache_dir,
    }
    if local_only:
        load_kwargs["local_files_only"] = True

    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model.eval()

    print(f"Loading tokenizer: {model_name}")
    tokenizer_kwargs = {"trust_remote_code": True, "cache_dir": cache_dir}
    if local_only:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available, using CPU (this will be slow)")

    return model, tokenizer


def eval_model(args):
    """Evaluate MiniCPM-V-4_5 model on SpatialRGPT-Bench with batch evaluation."""

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.cache_dir, local_only=args.local_only
    )

    # Get model name for output
    model_name = Path(args.model_path).name

    # Load benchmark annotations
    print(f"Loading annotations from: {args.annotation_file}")
    with open(args.annotation_file) as f:
        questions = json.load(f)

    print(f"Total questions: {len(questions)}")

    # Prepare output directory
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Batch evaluation
    batch_size = args.batch_size
    total_batches = (len(questions) + batch_size - 1) // batch_size

    print(f"Batch size: {batch_size}, Total batches: {total_batches}")
    print(f"Saving results incrementally to: {answers_file}")

    # Open file in append mode and write results after each batch
    with open(answers_file, "w", encoding="utf-8") as f:
        pass  # Create/clear the file first

    for batch_idx in tqdm(range(total_batches), desc="Evaluating batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]

        # Prepare batch inputs
        batch_results = []

        for line in batch_questions:
            image_id = line["id"]
            image_info = line["image_info"]
            question_text = line["text_q"]
            qa_info = line["qa_info"]
            conversations = line["conversations"]

            # Load image from som_images folder with annotations
            image_path = os.path.join(args.image_folder, 'som_images', image_id + '.png')
            image = Image.open(image_path).convert("RGB")

            # Get question and ground truth (one conversation = one turn)
            question = question_text
            ground_truth = conversations[1]["value"]

            # Prepare messages with system prompt - MiniCPM-V format
            # MiniCPM-V uses a list format with role and content
            msgs = [
                {
                    "role": "user",
                    "content": [SYSTEM_PROMPT, image, question_text]
                }
            ]

            # Generate response
            with torch.inference_mode():
                res = model.chat(
                    image=None,  # image is already in msgs
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else 0.0,
                    top_p=args.top_p if args.top_p else 1.0,
                    max_new_tokens=args.max_new_tokens,
                )

            result = {
                "question_id": image_id,
                "question": question_text,
                "pred": res.strip() if isinstance(res, str) else res[0].strip(),
                "gt": ground_truth,
            }
            batch_results.append(result)

        # Append batch results to file
        with open(answers_file, "a", encoding="utf-8") as f:
            for result in batch_results:
                f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        tqdm.write(f"  Saved {len(batch_results)} new results (total: {(batch_idx + 1) * batch_size})...")

    print(f"Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MiniCPM-V-4_5 models on SpatialRGPT-Bench"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="openbmb/MiniCPM-V-4_5",
        help="HuggingFace model name or path (default: openbmb/MiniCPM-V-4_5)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory to store downloaded models (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only use locally cached models, don't download"
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to the image folder containing benchmark images"
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        required=True,
        help="Path to the SpatialRGPT-Bench annotation JSON file"
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        required=True,
        help="Path to the output answers JSON file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation (default: 0.0, greedy)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p sampling parameter (default: None)"
    )

    args = parser.parse_args()

    eval_model(args)
