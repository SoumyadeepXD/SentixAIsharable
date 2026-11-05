"""Load a trained SentixAI model and predict sentiment without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sentixai_data import clean_and_tokenize, encode_tokens
from sentixai_model import build_model


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", default="outputs/artifacts")
    parser.add_argument("--model-name", default="bilstm_attention.pt")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--top-tokens", type=int, default=10)
    args = parser.parse_args()
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    checkpoint = load_checkpoint(model_dir / args.model_name, device)
    if "vocabulary" in checkpoint:
        vocabulary = checkpoint["vocabulary"]
    elif (model_dir / f"{Path(args.model_name).stem}_vocabulary.json").exists():
        vocabulary = json.loads((model_dir / f"{Path(args.model_name).stem}_vocabulary.json").read_text(encoding="utf-8"))
    else:
        vocabulary = json.loads((model_dir / "vocabulary.json").read_text(encoding="utf-8"))
    model_type = checkpoint.get("model_type", checkpoint.get("model", "bilstm_attention"))
    model_config = dict(checkpoint.get("model_config", checkpoint.get("config", {})))
    vocab_size = model_config.pop("vocab_size", len(vocabulary))
    num_classes = model_config.pop("num_classes", len(checkpoint["class_names"]))
    model = build_model(model_type, vocab_size=vocab_size, num_classes=num_classes, **model_config).to(device)
    state_dict = checkpoint.get("model_state", checkpoint.get("state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    tokens = clean_and_tokenize(args.text)
    ids = encode_tokens(tokens, vocabulary, checkpoint["max_length"])
    with torch.no_grad():
        logits, weights = model(torch.tensor([ids], dtype=torch.long, device=device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        predicted_id = int(logits.argmax(dim=1).item())
        token_weights = weights.squeeze(0).cpu().tolist()[:len(tokens)]
    ranked_tokens = sorted(zip(tokens[:len(token_weights)], token_weights), key=lambda item: item[1], reverse=True)
    output = {
        "sentiment": checkpoint["class_names"][predicted_id],
        "probabilities": dict(zip(checkpoint["class_names"], probabilities)),
        "attention_tokens": [{"token": token, "weight": float(weight)} for token, weight in ranked_tokens[:args.top_tokens]],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
