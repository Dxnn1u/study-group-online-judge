from datasets import load_dataset
from collections import defaultdict
from transformers import AutoTokenizer
from labs.lab1 import GPT2, GPT2Config, load_weights
import torch
import hashlib
import json

REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"

# 3 prompt formatting
LETTERS = "ABCD"

def format_questions(row, answer=None):
    options = " ".join(f"({letter}) {choice}" for letter, choice in zip(LETTERS, row["choices"]))
    return f"{row['question']}\n{options}\nAnswer: {answer or ''}"

def build_prompt(row, examples):
    subject = row["subject"].replace("_", " ")
    shots = [format_questions(ex, LETTERS[ex["answer"]]) for ex in examples]
    return (
        f"The following are multiple choice questions about {subject}.\n\n"
        + "\n\n".join([*shots, format_questions(row)])
    )

# 5 Load model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_weights(GPT2(GPT2Config())).eval()
    model = model.to(device) # move model weights to device(GPU)
    if device.type == "cuda":
        model = model.half() # fp32 -> fp16
    return model, device

# 5 Batch evaluation
@torch.no_grad()
def predict_all(model, device, tokenizer, test, exemplars, letter_ids, batch_size=16):
    predictions = {}

    for start in range(0, len(test), batch_size):
        indices = range(start, min(start + batch_size, len(test)))
        rows = [test[i] for i in indices]

        prompts = [build_prompt(row, exemplars[row["subject"]]) for row in rows]
        
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        last = logits[:, -1, :]  # (batch_size, vocab_size)

        choice = last[:, letter_ids].argmax(dim=-1)  # (batch_size,)

        for i, row, c in zip(indices, rows, choice.tolist()):
            predictions[question_key(i, row)] = LETTERS[c]

    return predictions


#7 question key
def question_key(index, row):
    payload = {
        "index": index,
        "subject": row["subject"],
        "question": row["question"],
        "choices": row["choices"]
    }
    json_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(json_payload.encode("utf-8")).hexdigest()

def mmlu_eval(limit: int | None = None) -> dict[str, str]:
    """Return GPT-2's A/B/C/D prediction for every MMLU test question.

    Load ``cais/mmlu`` at revision
    ``c30699e8356da336a370243923dbaf21066bb9fe``. For each subject, use
    its first four ``dev`` questions as exemplars and evaluate its ``test``
    questions. Format the prompt as specified in the Lab 2 assignment. If a
    prompt exceeds GPT-2's context window, retain its final 1024 tokens.

    Each key is the SHA-256 of a compact UTF-8 JSON object with keys
    ``index``, ``subject``, ``question``, and ``choices`` (sorted keys,
    ``ensure_ascii=False``, compact separators). ``index`` is the zero-based
    row number of the pinned ``all`` test split. This disambiguates repeated
    questions, including 27 identical subject/question/choice rows. Each
    value is one of A/B/C/D, selected from the corresponding next-token
    logits. No question labels should be used to choose a prediction.
    """  

    # 1 setting
    dev = load_dataset("cais/mmlu", "all", revision=REVISION, split="dev")
    test = load_dataset("cais/mmlu", "all", revision=REVISION, split="test")

    if limit is not None:
        test = test.select(range(limit))

    # 2 exemplars
    exemplars = defaultdict(list)
    for row in dev:
        if len(exemplars[row["subject"]]) < 4:
            exemplars[row["subject"]].append(row)   
    # 4 tokenization
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left" # when exceeding context window, keep the last 1024 tokens

    letter_ids = [tokenizer.encode(letter)[0] for letter in LETTERS]

    model, device = load_model()

    predictions = predict_all(model, device, tokenizer, test, exemplars, letter_ids)

    return predictions
