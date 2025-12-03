import argparse
import os
from typing import Dict, List

import torch
import yaml

from lpzero.datasets.distributed_utils.data_utils import get_lm_corpus
from lpzero.model.model_loader import load_model_from_config
from lpzero.predictor.measures.lpzero import get_batch_data, get_lpzero


def _load_lora_space(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError("Expected lora_configs yaml to be a list or dict of configs")


def main():
    parser = argparse.ArgumentParser(description="Demo: LPZero scoring for LoRA configurations")
    parser.add_argument("--base_config", type=str, required=True, help="Path to the base GPT-style YAML config")
    parser.add_argument("--lora_configs", type=str, required=True, help="YAML file describing LoRA search space")
    parser.add_argument("--output", type=str, default="./lora_lpzero_scores.yaml", help="Where to store the scores")
    parser.add_argument("--model_type", type=str, default="hf_gpt2_flex", help="Model type registered in model_loader")
    parser.add_argument("--dataset", type=str, default="wt2", help="Dataset name (for get_lm_corpus)")
    parser.add_argument("--data", type=str, default="./data/wikitext/wikitext-2", help="Path to dataset root")
    parser.add_argument("--cache_dir", type=str, default="./data/cachedir", help="Cache directory for tokenized data")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the zero-cost proxy on")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size for a single LPZero evaluation batch")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length for LPZero eval")
    args = parser.parse_args()

    vocab = "gpt2" if "gpt" in args.model_type else "word"
    vocab_size = 50264 if "gpt" in args.model_type else None

    corpus = get_lm_corpus(args.data, args.cache_dir, args.dataset, vocab, vocab_size, refresh_cache=False)
    train_itr = corpus.get_iterator(
        "train", args.batch_size, args.seq_len, device=args.device, mem_len=0, ext_len=0
    )
    n_token = len(corpus.vocab)
    inputs, targets = get_batch_data(train_itr, 1)
    inputs, targets = inputs.to(args.device), targets.to(args.device)

    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)
    base_config["n_token"] = base_config.get("n_token", n_token)

    scores = {}
    for idx, lora_cfg in enumerate(_load_lora_space(args.lora_configs)):
        lora_name = lora_cfg.get("name", f"lora_{idx}")
        config = dict(base_config)
        config["lora_config"] = lora_cfg

        model = load_model_from_config(args.model_type, config)
        model.n_token = config["n_token"]
        model.to(args.device)
        with torch.no_grad():
            zc_score = get_lpzero(model, inputs, targets)
        scores[lora_name] = float(zc_score)
        print(f"{lora_name}: lpzero score {zc_score}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    with open(args.output, "w") as f:
        yaml.safe_dump(scores, f)
    print(f"Saved scores to {args.output}")


if __name__ == "__main__":
    main()
