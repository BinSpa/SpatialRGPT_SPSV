#!/usr/bin/env python3
"""
Rule-Based GT Evaluation for SpatialRGPT-Bench (v4)

No LLM canonicalization — uses regex for quantitative/direction GT,
and a single-pass LLM judge for qualitative answers.

Outputs:
  detailed_results_v4.json  — per-sample results
  summary_v4.json           — accuracy metrics
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from openai import OpenAI

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_BASE = "https://api.deepseek.com"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(SCRIPT_DIR, "..", "SpatialRGPT-Bench", "SpatialRGPT-Bench_v1.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results_file(filepath):
    with open(filepath) as f:
        content = f.read().strip()
    if content.startswith("["):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    data, decoder, idx = [], json.JSONDecoder(), 0
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
            data.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return data


def call_llm(client, model, prompt, max_tokens=128, max_retries=5,
             system="You are a precise evaluation assistant. Answer YES or NO only."):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
            else:
                print(f"  API failed after {max_retries} retries: {e}")
                return None


def parse_yes_no(content):
    if not content:
        return None
    lower = content.lower().strip()
    for line in reversed(lower.split("\n")):
        line = line.strip().rstrip(".! ")
        if line == "yes":
            return True
        if line == "no":
            return False
    matches = re.findall(r"\b(yes|no)\b", lower)
    return matches[-1] == "yes" if matches else None


def infer_type(qid):
    if qid.startswith("qualitative_"):
        return "qualitative"
    if qid.startswith("quantitative_"):
        return "quantitative"
    return "unknown"


# ---------------------------------------------------------------------------
# Rule-based GT extraction
# ---------------------------------------------------------------------------

UNIT_TO_METERS = {
    "meter": 1, "meters": 1, "m": 1,
    "centimeter": 0.01, "centimeters": 0.01, "cm": 0.01,
    "millimeter": 0.001, "millimeters": 0.001, "mm": 0.001,
    "inch": 0.0254, "inches": 0.0254, "in": 0.0254,
    "foot": 0.3048, "feet": 0.3048, "ft": 0.3048,
}

# Sort by length descending to match longest unit first
UNIT_PATTERN = "|".join(
    f"{k}" for k in sorted(UNIT_TO_METERS.keys(), key=len, reverse=True)
)

# Match: number + unit (unit must have a word boundary after)
MEASURE_RE = re.compile(
    rf"(\d+\.?\d*)\s*({UNIT_PATTERN})\b", re.I
)


def extract_measurement(gt_text):
    """Extract value in meters from GT text using regex."""
    match = MEASURE_RE.search(gt_text)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value * UNIT_TO_METERS.get(unit[:3], 1), match.group(0)
    return None, None


def extract_direction(gt_text):
    """Extract clock position from GT text."""
    match = re.search(r"(\d+)\s*o\.?\s*clock", gt_text, re.I)
    if match:
        return int(match.group(1)), match.group(0)
    return None, None


# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

EVAL_QUAL_PROMPT = """\
Determine if the model prediction is correct given the question and ground truth.

EVALUATION RULES — read the ground truth carefully:
- If GT starts with "Incorrect", "No,", "No." → the GT is saying the question's claim is WRONG. \
The model should AGREE with the negation.
- If GT starts with "Yes,", "Correct,", "Indeed,", "Actually," → the GT is affirming the relationship. \
The model should AGREE.
- If GT starts with "Region [X] is not..." → the negated relationship is TRUE.
- Oblique phrasing: "With less height is Region [1]" → Region [1] is shorter.
- "Positioned higher is Region [0]" → Region [0] is above.
- The core spatial relationship is what matters, not exact wording.

Question: {question}
Ground Truth: {gt}
Prediction: {pred}

Output ONLY YES or NO."""


EVAL_QUANT_PROMPT = """\
Extract the numeric measurement from this model prediction.

Convert to meters: 1 inch = 0.0254, 1 foot = 0.3048, 1 cm = 0.01, 1 mm = 0.001.
If no clear measurement is given, output null.

Question: {question}
Prediction: {pred}

Output JSON: {{"value_meters": <float or null>, "raw": "<number> <unit>"}}"""


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(data, bench_annotations, client, model, max_workers=8):
    results = []
    stats = {
        "qualitative": {"total": 0, "correct": 0},
        "quantitative": {"total": 0, "correct": 0},
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0}),
    }

    qual, quant_nd, quant_dir = [], [], []
    for item in data:
        qid = item.get("question_id", "")
        qtype = infer_type(qid)
        gt = item.get("gt", "")
        ann = bench_annotations.get(qid, {})
        cat = ann.get("category", "")

        sample = {
            "qid": qid, "question": item.get("question", ""),
            "pred": item.get("pred", ""), "gt": gt,
            "category": cat, "type": qtype,
        }

        if qtype == "qualitative":
            qual.append(sample)
        elif cat == "direction":
            quant_dir.append(sample)
        elif qtype == "quantitative":
            quant_nd.append(sample)

    # --- Qualitative: LLM judges agreement with raw GT ---
    print(f"\nEvaluating {len(qual)} qualitative samples...")

    def _eval_qual(s):
        prompt = EVAL_QUAL_PROMPT.format(
            question=s["question"], gt=s["gt"][:500], pred=s["pred"][:500],
        )
        content = call_llm(client, model, prompt, max_tokens=128,
                            system="You are a precise evaluation assistant. Answer YES or NO only.")
        return s["qid"], parse_yes_no(content)

    qual_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_eval_qual, s): s["qid"] for s in qual}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Qual"):
            qid, v = fut.result()
            qual_map[qid] = v

    for s in qual:
        ok = qual_map.get(s["qid"], False)
        if ok is None:
            ok = False
        results.append({
            "question_id": s["qid"], "question_type": "qualitative",
            "category": s["category"], "is_correct": ok,
            "question": s["question"], "pred": s["pred"], "gt": s["gt"],
        })
        stats["qualitative"]["total"] += 1
        stats["qualitative"]["correct"] += int(ok)
        stats["by_category"][s["category"]]["total"] += 1
        stats["by_category"][s["category"]]["correct"] += int(ok)

    # --- Quantitative (non-direction): regex GT + LLM/regex pred extraction ---
    print(f"\nEvaluating {len(quant_nd)} quantitative samples...")

    # Extract GT values via regex (no API calls)
    for s in quant_nd:
        gt_m, gt_raw = extract_measurement(s["gt"])
        s["gt_meters"] = gt_m
        s["gt_raw"] = gt_raw

    gt_count = sum(1 for s in quant_nd if s["gt_meters"] is not None)
    print(f"  GT extraction: {gt_count}/{len(quant_nd)} have measurable values")

    def _eval_quant(s):
        prompt = EVAL_QUANT_PROMPT.format(
            question=s["question"], pred=s["pred"][:500],
        )
        content = call_llm(client, model, prompt, max_tokens=128)
        pred_m = None
        parsed = None
        if content:
            try:
                if "```json" in content:
                    content = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL).group(1)
                elif "```" in content:
                    content = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL).group(1)
                parsed = json.loads(content.strip())
            except Exception:
                parsed = None
        if parsed:
            pred_m = parsed.get("value_meters")
        # Regex fallback on pred
        if pred_m is None:
            m = MEASURE_RE.search(s["pred"])
            if m:
                val = float(m.group(1))
                u = m.group(2).lower()
                pred_m = val * UNIT_TO_METERS.get(u[:3], 1)
        gt_m = s["gt_meters"]
        ok = False
        if pred_m is not None and gt_m is not None and gt_m != 0:
            ok = abs(pred_m - gt_m) / gt_m <= 0.25
        return s["qid"], ok, pred_m, gt_m

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_eval_quant, s): s["qid"] for s in quant_nd}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Quant"):
            qid, ok, pm, gm = fut.result()
            s = next(s for s in quant_nd if s["qid"] == qid)
            results.append({
                "question_id": qid, "question_type": "quantitative",
                "category": "distance_data", "is_correct": ok,
                "pred_meters": pm, "gt_meters": gm,
                "gt_raw": s.get("gt_raw"),
            })
            stats["quantitative"]["total"] += 1
            stats["quantitative"]["correct"] += int(ok)
            stats["by_category"]["distance_data"]["total"] += 1
            stats["by_category"]["distance_data"]["correct"] += int(ok)

    # --- Direction: regex GT + LLM judge ---
    print(f"\nEvaluating {len(quant_dir)} direction samples...")

    def _eval_dir(s):
        gt_clock, gt_raw = extract_direction(s["gt"])
        prompt = EVAL_QUAL_PROMPT.format(
            question=s["question"],
            gt=f"Direction: {gt_raw}" if gt_raw else s["gt"],
            pred=s["pred"][:500],
        )
        content = call_llm(client, model, prompt, max_tokens=128,
                            system="You are a precise evaluation assistant. Answer YES or NO only.")
        return s["qid"], parse_yes_no(content), gt_clock

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_eval_dir, s): s["qid"] for s in quant_dir}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Direct"):
            qid, v, gc = fut.result()
            if v is None:
                v = False
            results.append({
                "question_id": qid, "question_type": "quantitative",
                "category": "direction", "is_correct": v,
                "gt_clock": gc,
            })
            stats["quantitative"]["total"] += 1
            stats["quantitative"]["correct"] += int(v)
            stats["by_category"]["direction"]["total"] += 1
            stats["by_category"]["direction"]["correct"] += int(v)

    results.sort(key=lambda x: x.get("question_id", ""))
    return results, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Rule-Based GT Evaluation for SpatialRGPT-Bench (v4)")
    ap.add_argument("input_file", help="Path to model results file")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api_base", default=DEFAULT_API_BASE)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--max_workers", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.input_file)
    os.makedirs(output_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: No API key. Set DEEPSEEK_API_KEY or --api_key")
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=args.api_base)

    # Load benchmark annotations for categories
    bench_annotations = {}
    with open(BENCH_PATH) as f:
        annotations = json.load(f)
    for a in annotations:
        bench_annotations[a["id"]] = a.get("qa_info", {})

    # Load model results
    print(f"Loading results from: {args.input_file}")
    data = load_results_file(args.input_file)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} samples")

    results, stats = evaluate(data, bench_annotations, client, args.model, args.max_workers)

    # Metrics
    qn, qq = stats["quantitative"], stats["qualitative"]
    total = qn["total"] + qq["total"]
    tc = qn["correct"] + qq["correct"]
    summary = {
        "total_accuracy": round(tc / total * 100, 2) if total else 0,
        "quantitative_accuracy": round(qn["correct"] / qn["total"] * 100, 2) if qn["total"] else 0,
        "qualitative_accuracy": round(qq["correct"] / qq["total"] * 100, 2) if qq["total"] else 0,
        "total_samples": total,
        "quantitative_samples": qn["total"], "quantitative_correct": qn["correct"],
        "qualitative_samples": qq["total"], "qualitative_correct": qq["correct"],
        "by_category": {},
    }
    for cat, cs in sorted(stats["by_category"].items()):
        summary["by_category"][cat] = {
            "accuracy": round(cs["correct"] / cs["total"] * 100, 2) if cs["total"] else 0,
            "correct": cs["correct"], "total": cs["total"],
        }

    # Save
    with open(os.path.join(output_dir, "detailed_results_v4.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "summary_v4.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print("RULE-BASED GT EVALUATION SUMMARY (v4)")
    print("=" * 55)
    print(f"Total:           {summary['total_accuracy']:.2f}%  ({tc}/{total})")
    print(f"  Qualitative:    {summary['qualitative_accuracy']:.2f}%  ({qq['correct']}/{qq['total']})")
    print(f"  Quantitative:   {summary['quantitative_accuracy']:.2f}%  ({qn['correct']}/{qn['total']})")
    print("=" * 55)
    for cat, cs in sorted(summary["by_category"].items()):
        print(f"  {cat:<30s} {cs['accuracy']:>6.2f}%  ({cs['correct']}/{cs['total']})")
    print("=" * 55)


if __name__ == "__main__":
    main()
