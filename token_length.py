import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize dataset and compute token lengths")
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

def build_input_fs(context, prompt, response):
    header = f"{INSTRUCT}\n"
    demonstrations = ""
    for ex in FEW_SHOTS:
        demonstrations += (
            f"Context: {ex['context']}\n"
            f"Prompt: {ex['prompt']}\n"
            f"Response: {ex['response']}\n"
            f"Label: {ex['label']}\n\n"
        )
    query = (
        f"Context: {context}\n"
        f"Prompt: {prompt}\n"
        f"Response: {response}\n"
        f"Label:"
    )
    return header + demonstrations + query

def build_input(context, prompt, response):
    header = f"{INSTRUCT}\n"
    query = (
        f"Context: {context}\n"
        f"Prompt: {prompt}\n"
        f"Response: {response}\n"
        f"Label:"
    )
    return header + query


def main():
    args = parse_args()

    print("Loading evaluation data...")
    df_eval = pd.read_csv(args.eval_path)

    # Drop old token_length if exists
    if "token_length" in df_eval.columns:
        df_eval = df_eval.drop("token_length", axis=1)

    dataset_eval = Dataset.from_pandas(df_eval, preserve_index=False)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added new pad token.")

    print("Building text inputs...")
    if args.fewshot:
        texts = [build_input_fs(c, p, r) for c, p, r in zip(df_eval["context"], df_eval["prompt"], df_eval["response"])]
    else:
        texts = [build_input(c, p, r) for c, p, r in zip(df_eval["context"], df_eval["prompt"], df_eval["response"])]

    print("Tokenizing...")
    tokenized = tokenizer(texts, truncation=True, padding=False)

    # Compute token lengths
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    df_eval["token_length"] = lengths

    # Save dataset with token length
    df_eval.to_csv(args.output_csv, index=False)
    print(f"Saved tokenized dataset with lengths to {args.output_csv}")

    # Statistics
    arr = np.array(lengths)
    stats = {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr))
    }
    print("Token length statistics:", stats)


if __name__ == "__main__":
    main()
