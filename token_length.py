import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize dataset and compute token lengths per field")
    parser.add_argument("--base_model_dir", type=str, required=True, help="Path to base model directory")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to evaluation CSV")
    parser.add_argument("--output_csv", type=str, default="./tokenized_dataset.csv", help="Where to save dataset with token length")
    parser.add_argument("--fewshot", action="store_true", help="Enable few-shot mode")
    return parser.parse_args()


INSTRUCT = "You are given a context, a user question, and a model answer. Determine whether the answer is correct, intrinsic hallucination, or extrinsic hallucination."
FEW_SHOTS = [
    {
        "context": "Paris is the capital of France.",
        "prompt": "What is the capital of France?",
        "response": "Paris is the capital.",
        "label": "no"
    },
    {
        "context": "Paris is the capital of France.",
        "prompt": "What is the capital of France?",
        "response": "Marseille is the capital.",
        "label": "intrinsic"
    },
    {
        "context": "Paris is the capital of France.",
        "prompt": "What     is the capital of France?",
        "response": "Paris is the capital, and it has 20 million people.",
        "label": "extrinsic"
    },
]

def build_fs_prefix():
    """Xây dựng phần few-shot demonstration, không gộp với input chính"""
    header = f"{INSTRUCT}\n"
    demonstrations = ""
    for ex in FEW_SHOTS:
        demonstrations += (
            f"Context: {ex['context']}\n"
            f"Prompt: {ex['prompt']}\n"
            f"Response: {ex['response']}\n"
            f"Label: {ex['label']}\n\n"
        )
    return header + demonstrations


def main():
    args = parse_args()

    print("Loading evaluation data...")
    df_eval = pd.read_csv(args.eval_path)

    # Drop old token_length if exists
    for col in ["context_len", "prompt_len", "response_len"]:
        if col in df_eval.columns:
            df_eval = df_eval.drop(col, axis=1)

    dataset_eval = Dataset.from_pandas(df_eval, preserve_index=False)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added new pad token.")

    # Build fewshot prefix if needed
    fs_prefix = build_fs_prefix() if args.fewshot else ""

    print("Tokenizing fields separately...")
    context_texts = [fs_prefix + "Context: " + str(c) for c in df_eval["context"]]
    prompt_texts  = [fs_prefix + "Prompt: "  + str(p) for p in df_eval["prompt"]]
    response_texts= [fs_prefix + "Response: "+ str(r) for r in df_eval["response"]]

    context_tokens  = tokenizer(context_texts, truncation=True, padding=False)["input_ids"]
    prompt_tokens   = tokenizer(prompt_texts, truncation=True, padding=False)["input_ids"]
    response_tokens = tokenizer(response_texts, truncation=True, padding=False)["input_ids"]

    # Save lengths into dataframe
    df_eval["context_len"]  = [len(ids) for ids in context_tokens]
    df_eval["prompt_len"]   = [len(ids) for ids in prompt_tokens]
    df_eval["response_len"] = [len(ids) for ids in response_tokens]

    # Save dataset with token length
    df_eval.to_csv(args.output_csv, index=False)
    print(f"Saved tokenized dataset with lengths to {args.output_csv}")

    # Statistics
    stats = {
        "context": {
            "min": int(np.min(df_eval["context_len"])),
            "max": int(np.max(df_eval["context_len"])),
            "mean": float(np.mean(df_eval["context_len"])),
            "median": float(np.median(df_eval["context_len"]))
        },
        "prompt": {
            "min": int(np.min(df_eval["prompt_len"])),
            "max": int(np.max(df_eval["prompt_len"])),
            "mean": float(np.mean(df_eval["prompt_len"])),
            "median": float(np.median(df_eval["prompt_len"]))
        },
        "response": {
            "min": int(np.min(df_eval["response_len"])),
            "max": int(np.max(df_eval["response_len"])),
            "mean": float(np.mean(df_eval["response_len"])),
            "median": float(np.median(df_eval["response_len"]))
        }
    }
    print("Token length statistics per field:", stats)


if __name__ == "__main__":
    main()