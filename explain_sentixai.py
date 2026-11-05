"""Generate actual attention, LIME, and/or SHAP explanations from a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentixai_inference import SavedTextClassifier
from sentixai_xai import attention_explanation, lime_explanation, shap_explanation

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", choices=["sentiment", "emotion"], default="sentiment")
    parser.add_argument("--model-dir", default=None, help="Directory containing the model checkpoint and vocabulary.json")
    parser.add_argument("--model-name", default=None, help="Checkpoint filename (e.g. textcnn.pt)")
    parser.add_argument("--methods", nargs="+", choices=["attention", "lime", "shap"], default=["attention"])
    parser.add_argument("--lime-samples", type=int, default=500)
    parser.add_argument("--shap-samples", type=int, default=160)
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
        default_name = "bilstm_attention.pt" if args.model == "sentiment" else "emotion_bilstm_attention.pt"
        model_name = args.model_name or default_name
    elif args.model == "sentiment":
        model_dir = ROOT / "outputs/artifacts"
        model_name = args.model_name or "bilstm_attention.pt"
    else:
        model_dir = ROOT / "outputs/emotion/artifacts"
        model_name = args.model_name or "emotion_bilstm_attention.pt"

    classifier = SavedTextClassifier(model_dir, model_name)
    prediction = classifier.predict(args.text)
    explanations = {}
    if "attention" in args.methods:
        explanations["attention"] = attention_explanation(prediction)
    if "lime" in args.methods:
        explanations["lime"] = lime_explanation(classifier, args.text, num_samples=args.lime_samples)
    if "shap" in args.methods:
        explanations["shap"] = shap_explanation(classifier, args.text, nsamples=args.shap_samples)
    print(json.dumps({"prediction": prediction, "explanation": explanations}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
