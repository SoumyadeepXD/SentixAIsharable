"""Script to evaluate trained models on 12 diverse sentiment review samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from sentixai_data import clean_and_tokenize, encode_tokens
from sentixai_inference import SavedTextClassifier
from sentixai_model import build_model
from sentixai_xai import attention_explanation, lime_explanation

SAMPLE_TEXTS = [
    # 1. Strongly positive
    ("An absolute triumph of cinema, featuring mesmerizing performances and stunning visuals.", "positive"),
    # 2. Strongly negative
    ("A dreadful and agonizingly slow trainwreck that completely insults the intelligence of the audience.", "negative"),
    # 3. Mixed positive
    ("While the pacing faltered somewhat in the middle act, the heartfelt ending made it a deeply rewarding experience.", "positive"),
    # 4. Mixed negative
    ("The cinematography was gorgeous, but the terrible script and wooden acting ruined the entire movie.", "negative"),
    # 5. Short positive
    ("Brilliant, witty, and thoroughly enjoyable.", "positive"),
    # 6. Short negative
    ("Utter garbage and a total waste of money.", "negative"),
    # 7. Sarcastic / subtle
    ("I have never seen a movie make two hours feel quite so much like two centuries.", "negative"),
    # 8. High contrast
    ("Started off promising, but devolved into sheer stupidity and chaos by the second half.", "negative"),
    # 9. Masterpiece praise
    ("One of the finest cinematic achievements of our time, truly unforgettable and poignant.", "positive"),
    # 10. Unwatchable disaster
    ("Horrendous directing, awful writing, and easily the worst film of the year.", "negative"),
    # 11. Technical praise
    ("The incredible sound design and exceptional musical score elevated an otherwise ordinary plot.", "positive"),
    # 12. Emotional resonance
    ("Heartbreaking and powerful acting that moved the entire audience to tears of joy and empathy.", "positive"),
]


def test_samples_on_classifier(classifier: SavedTextClassifier, name: str = "Classifier") -> None:
    print(f"\n{'=' * 75}")
    print(f"EVALUATING 12 DIVERSE SAMPLES ON: {name}")
    print(f"{'=' * 75}")

    correct = 0
    total = len(SAMPLE_TEXTS)

    for i, (text, expected) in enumerate(SAMPLE_TEXTS, start=1):
        pred = classifier.predict(text)
        predicted_label = pred["label"]
        confidence = pred["confidence"]
        top_tokens = [f"{t['token']}:{t['weight']:.3f}" for t in pred["attention_tokens"][:4]]
        is_correct = (predicted_label == expected)
        if is_correct:
            correct += 1
        mark = "✓ PASS" if is_correct else "✗ FAIL"

        print(f"[{i:02d}] {mark} | Exp: {expected:8s} | Got: {predicted_label:8s} ({confidence*100:.1f}%) | Text: \"{text[:55]}...\"")
        print(f"     Top Attention Tokens: {', '.join(top_tokens)}")

    accuracy = (correct / total) * 100.0
    print(f"\nAccuracy across sample suite: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'=' * 75}\n")


def test_experiment_model(exp_dir: Path, vocab_path: Path) -> None:
    ckpt_path = exp_dir / "model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    model_type = checkpoint.get("model", "bilstm_attention")
    model_config = checkpoint.get("config", {})
    model = build_model(model_type, vocab_size=len(vocab), num_classes=len(checkpoint["class_names"]), **model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print(f"\n{'=' * 75}")
    print(f"EVALUATING 12 SAMPLES ON EXPERIMENT: {exp_dir.name} ({model_type})")
    print(f"{'=' * 75}")

    correct = 0
    total = len(SAMPLE_TEXTS)
    for i, (text, expected) in enumerate(SAMPLE_TEXTS, start=1):
        tokens = clean_and_tokenize(text)
        encoded = encode_tokens(tokens, vocab, 200)
        with torch.no_grad():
            logits, weights = model(torch.tensor([encoded], dtype=torch.long))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
            pred_id = int(logits.argmax(dim=1).item())
            attn = weights.squeeze(0).cpu().tolist()[:len(tokens)]

        predicted_label = checkpoint["class_names"][pred_id]
        confidence = probs[pred_id]
        ranked_attn = sorted(zip(tokens, attn), key=lambda x: x[1], reverse=True)[:4]
        top_tokens = [f"{t}:{w:.3f}" for t, w in ranked_attn]

        is_correct = (predicted_label == expected)
        if is_correct:
            correct += 1
        mark = "✓ PASS" if is_correct else "✗ FAIL"

        print(f"[{i:02d}] {mark} | Exp: {expected:8s} | Got: {predicted_label:8s} ({confidence*100:.1f}%) | Text: \"{text[:55]}...\"")
        print(f"     Top Attention Tokens: {', '.join(top_tokens)}")

    accuracy = (correct / total) * 100.0
    print(f"\nAccuracy across sample suite: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'=' * 75}\n")


if __name__ == "__main__":
    art_dir = ROOT / "outputs/artifacts"

    # 1. Test on production BiLSTM checkpoint
    print("\n--- 1. Testing Production BiLSTM + Attention ---")
    bilstm_clf = SavedTextClassifier(art_dir, "bilstm_attention.pt")
    test_samples_on_classifier(bilstm_clf, "BiLSTM + Attention (Production)")

    # 2. Test on trained TextCNN (Attention Pooling)
    if (art_dir / "textcnn.pt").exists():
        print("\n--- 2. Testing TextCNN (Attention Pooling) ---")
        cnn_clf = SavedTextClassifier(art_dir, "textcnn.pt")
        test_samples_on_classifier(cnn_clf, "TextCNN Attention (outputs/artifacts/textcnn.pt)")

    # 3. Test on trained TextCNN (Max Pooling)
    if (art_dir / "textcnn_max.pt").exists():
        print("\n--- 3. Testing TextCNN (Max Pooling) ---")
        cnn_max_clf = SavedTextClassifier(art_dir, "textcnn_max.pt")
        test_samples_on_classifier(cnn_max_clf, "TextCNN Max Pooling (outputs/artifacts/textcnn_max.pt)")
