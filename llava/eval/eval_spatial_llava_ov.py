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
# Evaluation script for LLaVA-OV-1.5 models on SpatialRGPT-Bench
# Based on official LLaVA-OneVision usage: https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct

import argparse
import json
import os
import re
import ast
from pathlib import Path
import subprocess

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


# Default cache directory for models
DEFAULT_CACHE_DIR = "/root/autodl-fs/models"


# System prompt for SpatialRGPT-Bench evaluation
SYSTEM_PROMPT = """You are an AI assistant for spatial reasoning evaluation.
You will be shown an image and asked a question about spatial relationships between objects or regions in the image.
The Regions in question have been annotated with their outlines and a number in the image.

Please follow these rules:
1. Answer the question directly and concisely based on what you see in the image.
2. For comparison questions (bigger/smaller, left/right, etc.), clearly state which object/region has which property.
3. For measurement questions, provide your best estimate with specific metric or imperial units based on the visual information.

Your responses will be evaluated for accuracy against ground truth answers."""


def check_model_downloaded(model_name, cache_dir):
    """Check if model is already downloaded to cache directory."""
    # Convert model name to directory format used by huggingface
    model_dir_name = model_name.replace("/", "--")
    model_path = os.path.join(cache_dir, f"models--{model_dir_name}")
    
    if os.path.exists(model_path):
        # Check if model has required files
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                return True, os.path.join(model_path, "snapshots", snapshots[0])
    return False, None


def download_model(model_name, cache_dir):
    """Download model from HuggingFace using huggingface-cli."""
    print(f"Model '{model_name}' not found in cache. Downloading...")
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Use huggingface-cli to download the model
        subprocess.run(
            ["huggingface-cli", "download", model_name, "--local-dir", cache_dir],
            check=True
        )
        print(f"Model '{model_name}' downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        return False
    except FileNotFoundError:
        print("huggingface-cli not found. Installing huggingface_hub...")
        subprocess.run(["pip", "install", "huggingface_hub"], check=True)
        return download_model(model_name, cache_dir)


def load_model_and_processor(model_name, cache_dir):
    """Load model and processor, downloading if necessary."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    print(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    return model, processor


def eval_model(args):
    """Evaluate LLaVA-OV-1.5 model on SpatialRGPT-Bench with batch evaluation."""

    # Load model and processor (auto-download if needed)
    model, processor = load_model_and_processor(args.model_path, args.cache_dir)

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
        batch_messages = []
        batch_images = []
        batch_metadata = []

        for line in batch_questions:
            image_id = line["id"]
            image_info = line["image_info"]
            question_text = line["text_q"]
            qa_info = line["qa_info"]
            conversations = line["conversations"]

            # Load image
            image_path = os.path.join(args.image_folder, image_id + '.png')
            image = Image.open(image_path).convert("RGB")

            # Get question and ground truth (one conversation = one turn)
            question = question_text
            ground_truth = conversations[1]["value"]

            # Prepare messages with system prompt
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
            batch_images.append(image)
            batch_metadata.append({
                "question_id": image_id,
                "question": question_text,
                "gt": ground_truth,
                "qa_info": qa_info,
            })

        # Process batch
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        batch_texts = []
        for messages in batch_messages:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        # Process vision info for batch
        from qwen_vl_utils import process_vision_info
        batch_image_inputs = []
        batch_video_inputs = []
        for messages in batch_messages:
            image_inputs, video_inputs = process_vision_info(messages)
            batch_image_inputs.append(image_inputs)
            batch_video_inputs.append(video_inputs)

        # Process inputs for batch
        inputs = processor(
            text=batch_texts,
            images=batch_image_inputs,
            # videos=batch_video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)
        # Generate responses for batch
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

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode outputs
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Store results and save incrementally (only new batch results)
        batch_results = []
        for i, output_text in enumerate(output_texts):
            result = {
                "question_id": batch_metadata[i]["question_id"],
                "question": batch_metadata[i]["question"],
                "pred": output_text,
                "gt": batch_metadata[i]["gt"],
            }
            batch_results.append(result)
        
        # Append only the new batch results to file
        with open(answers_file, "a", encoding="utf-8") as f:
            for result in batch_results:
                f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        tqdm.write(f"  Saved {len(batch_results)} new results (total: {(batch_idx + 1) * batch_size})...")

    print(f"Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLaVA-OV-1.5 models on SpatialRGPT-Bench"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., lmms-lab/LLaVA-OneVision-1.5-8B-Instruct)"
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
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128)"
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
