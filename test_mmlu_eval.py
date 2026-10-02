"""Local checks for src/labs/lab2.py, mirroring the study-group judge.

The reference logic below is copied from the judge's Lab 2 task
(src/judge/tasks/lab2.py) so this file does not need the judge package.

Usage (from the repository root):
    python test_mmlu_eval.py                 # prompt/key checks + first 64 questions
    python test_mmlu_eval.py --limit 256     # more questions
    python test_mmlu_eval.py --all           # every test question (use a GPU, e.g. on Nano4)

Checks:
  1. build_prompt matches the judge's prompt for every test question (no model needed)
  2. question_key matches the judge's key for every test question (no model needed)
  3. mmlu_eval predictions agree with an HF GPT-2 reference on the evaluated questions
     (the judge requires >= 97% agreement)

For step 3, mmlu_eval should accept an optional ``limit`` so only the first N
test questions are evaluated; the judge calls it with no arguments.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from labs import lab2  # noqa: E402

DATASET_ID = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MODEL_ID = "openai-community/gpt2"
LETTERS = "ABCD"
MIN_AGREEMENT_PERCENT = 97

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---- Judge reference logic (copied from src/judge/tasks/lab2.py) ----

def judge_question_key(index, row):
    value = {
        "index": index,
        "subject": row["subject"],
        "question": row["question"],
        "choices": row["choices"],
    }
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def judge_format_question(row, answer=None):
    options = " ".join(
        f"({letter}) {choice}" for letter, choice in zip(LETTERS, row["choices"], strict=True)
    )
    return f"{row['question']}\n{options}\nAnswer: {answer or ''}"


def judge_prompt(row, exemplars):
    subject = row["subject"].replace("_", " ")
    examples = [judge_format_question(ex, LETTERS[ex["answer"]]) for ex in exemplars]
    return (
        f"The following are multiple choice questions about {subject}.\n\n"
        + "\n\n".join([*examples, judge_format_question(row)])
    )


@torch.no_grad()
def judge_reference(rows, exemplars, batch_size=16):
    """HF GPT-2 predictions for `rows` (indices are positions in the full test split)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The judge uses fp16 on CUDA; torch 2.2 on CPU lacks fp16 LayerNorm, so use fp32 there.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    print(f"  reference: HF GPT-2 on {device} in {dtype}")

    predictions = {}
    for start in range(0, len(rows), batch_size):
        batch = range(start, min(start + batch_size, len(rows)))
        prompts = [judge_prompt(rows[i], exemplars[rows[i]["subject"]]) for i in batch]
        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=model.config.n_positions,
            return_tensors="pt",
        ).to(device)
        positions = (enc["attention_mask"].cumsum(dim=1) - 1).clamp_min(0)
        logits = model(**enc, position_ids=positions).logits[:, -1, :]
        choices = logits[:, letter_ids].argmax(dim=-1).tolist()
        for i, c in zip(batch, choices):
            predictions[judge_question_key(i, rows[i])] = LETTERS[c]
    return predictions


# ---- Checks ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=64, help="number of test questions to predict")
    parser.add_argument("--all", action="store_true", help="predict every test question")
    args = parser.parse_args()

    dev = load_dataset(DATASET_ID, "all", split="dev", revision=DATASET_REVISION)
    test = load_dataset(DATASET_ID, "all", split="test", revision=DATASET_REVISION)
    exemplars = defaultdict(list)
    for row in dev:
        if len(exemplars[row["subject"]]) < 4:
            exemplars[row["subject"]].append(row)
    rows = list(test)
    print(f"loaded {len(rows)} test questions, {len(exemplars)} subjects\n")

    # 1. Prompts, character for character.
    bad = [
        i for i, row in enumerate(rows)
        if lab2.build_prompt(row, exemplars[row["subject"]]) != judge_prompt(row, exemplars[row["subject"]])
    ]
    detail = f"{len(rows) - len(bad)}/{len(rows)} identical"
    if bad:
        i = bad[0]
        mine = lab2.build_prompt(rows[i], exemplars[rows[i]["subject"]])
        ref = judge_prompt(rows[i], exemplars[rows[i]["subject"]])
        pos = next((k for k, (a, b) in enumerate(zip(mine, ref)) if a != b), min(len(mine), len(ref)))
        detail += (f"; first difference in row {i} at char {pos}: "
                   f"mine={mine[max(0, pos - 30):pos + 30]!r} judge={ref[max(0, pos - 30):pos + 30]!r}")
    check("prompts match the judge", not bad, detail)

    # 2. Keys.
    bad_keys = sum(lab2.question_key(i, row) != judge_question_key(i, row) for i, row in enumerate(rows))
    check("question keys match the judge", bad_keys == 0, f"{len(rows) - bad_keys}/{len(rows)} identical")

    # 3. Predictions vs HF reference.
    n = len(rows) if args.all else min(args.limit, len(rows))
    print(f"\nevaluating the first {n} test questions")
    t0 = time.time()
    try:
        preds = lab2.mmlu_eval() if n == len(rows) else lab2.mmlu_eval(limit=n)
    except TypeError:
        print("  mmlu_eval() does not accept `limit`; running every question (slow on CPU)")
        preds = lab2.mmlu_eval()
    print(f"  mmlu_eval took {time.time() - t0:.1f}s")

    subset = rows[:n]
    expected_keys = {judge_question_key(i, row) for i, row in enumerate(subset)}
    ok_contract = (
        isinstance(preds, dict)
        and expected_keys <= set(preds)
        and all(preds[k] in LETTERS for k in expected_keys)
    )
    check("return contract (dict of A/B/C/D for every evaluated key)", ok_contract,
          f"{len(expected_keys & set(preds)) if isinstance(preds, dict) else 0}/{len(expected_keys)} keys present")
    if not ok_contract:
        print(f"\n{sum(results)}/{len(results)} checks passed")
        return

    ref = judge_reference(subset, exemplars)
    mismatched = defaultdict(list)
    for i, row in enumerate(subset):
        k = judge_question_key(i, row)
        if preds[k] != ref[k]:
            mismatched[row["subject"]].append((i, preds[k], ref[k]))
    n_bad = sum(len(v) for v in mismatched.values())
    agreement = (n - n_bad) / n
    for subject, items in sorted(mismatched.items()):
        examples = ", ".join(f"row {i}: got {g}, expected {w}" for i, g, w in items[:3])
        print(f"  {subject}: {len(items)} mismatched; {examples}")
    check(f"agreement with reference >= {MIN_AGREEMENT_PERCENT}%",
          agreement * 100 >= MIN_AGREEMENT_PERCENT, f"{n - n_bad}/{n} = {agreement:.2%}")

    correct = sum(preds[judge_question_key(i, row)] == LETTERS[row["answer"]] for i, row in enumerate(subset))
    print(f"  (info) MMLU accuracy on these questions: {correct / n:.2%}")

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
