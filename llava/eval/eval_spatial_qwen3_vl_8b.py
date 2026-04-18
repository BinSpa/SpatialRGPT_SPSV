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
# Evaluation script for Qwen3-VL-8B-Instruct models on SpatialRGPT-Bench
# Based on official Qwen3-VL usage: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
# NOTE: Requires transformers >= 5.5.3

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration


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
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )

    print(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    return model, processor


def eval_model(args):
    """Evaluate Qwen3-VL-8B model on SpatialRGPT-Bench with batch evaluation."""

    model, processor = load_model_and_processor(args.model_path, args.cache_dir)

    model_name = Path(args.model_path).name

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

        batch_messages = []
        batch_metadata = []

        for line in batch_questions:
            image_id = line["id"]
            question_text = line["text_q"]
            conversations = line["conversations"]
            ground_truth = conversations[1]["value"]

            image_path = os.path.join(args.image_folder, 'som_images', image_id + '.png')
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]

            batch_messages.append(messages)
            batch_metadata.append({
                "question_id": image_id,
                "question": question_text,
                "gt": ground_truth,
            })

        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        batch_texts = []
        for messages in batch_messages:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        batch_image_inputs = []
        for messages in batch_messages:
            image_inputs, video_inputs = process_vision_info(messages)
            batch_image_inputs.append(image_inputs)

        all_images = []
        for img_list in batch_image_inputs:
            all_images.extend(img_list)

        inputs = processor(
            text=batch_texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        batch_results = []
        for i, output_text in enumerate(output_texts):
            text = output_text.strip()
            if "</think" in text:
                text = text.split("</think")[-1].strip()

            result = {
                "question_id": batch_metadata[i]["question_id"],
                "question": batch_metadata[i]["question"],
                "pred": text,
                "gt": batch_metadata[i]["gt"],
            }
            batch_results.append(result)

        with open(answers_file, "a", encoding="utf-8") as f:
            for result in batch_results:
                f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        tqdm.write(f"  Saved {len(batch_results)} new results (total: {(batch_idx + 1) * batch_size})...")

    print(f"Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-8B-Instruct models on SpatialRGPT-Bench"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model name or path (default: Qwen/Qwen3-VL-8B-Instruct)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory to store downloaded models (default: {DEFAULT_CACHE_DIR})"
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
        default=4,
        help="Batch size for evaluation (default: 4)"
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
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search (default: 1)"
    )

    args = parser.parse_args()
    eval_model(args)
