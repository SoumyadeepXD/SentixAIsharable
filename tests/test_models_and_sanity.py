"""Comprehensive test and sanity verification for SentixAI models and inference.

Can be run via:
    python tests/test_models_and_sanity.py
or
    pytest tests/test_models_and_sanity.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sentixai_inference import SavedTextClassifier
from sentixai_model import BiLSTMAttentionClassifier, TextCNNClassifier, build_model
from sentixai_xai import attention_explanation, lime_explanation, shap_explanation


class TestModelSanity(unittest.TestCase):
    def test_output_shapes_and_weights_sum(self) -> None:
        for model_type in ["bilstm_attention", "textcnn", "textcnn_max"]:
            with self.subTest(model_type=model_type):
                vocab_size = 150
                num_classes = 3
                model = build_model(
                    model_type,
                    vocab_size=vocab_size,
                    num_classes=num_classes,
                    embedding_dim=32,
                    hidden_dim=32,
                    num_filters=16,
                    kernel_sizes=(3, 4, 5),
                )
                model.eval()

                # Batch of 2 rows, T=6 tokens
                # Row 0: 4 real tokens, 2 pad tokens (pad=0)
                # Row 1: 3 real tokens, 3 pad tokens
                batch = torch.tensor([
                    [10, 20, 30, 40, 0, 0],
                    [15, 25, 35, 0, 0, 0],
                ], dtype=torch.long)

                with torch.no_grad():
                    logits, weights = model(batch)

                # 1. Output shapes
                self.assertEqual(logits.shape, (2, num_classes))
                self.assertEqual(weights.shape, (2, 6))

                # 2. Token weights sum to ~1 over non-pad tokens and ~0 on pad tokens
                weights_np = weights.cpu().numpy()
                self.assertAlmostEqual(float(weights_np[0, :4].sum()), 1.0, places=3)
                self.assertTrue((weights_np[0, 4:] < 1e-4).all())

                self.assertAlmostEqual(float(weights_np[1, :3].sum()), 1.0, places=3)
                self.assertTrue((weights_np[1, 3:] < 1e-4).all())

    def test_no_nan_on_all_padding_row(self) -> None:
        for model_type in ["bilstm_attention", "textcnn", "textcnn_max"]:
            with self.subTest(model_type=model_type):
                model = build_model(
                    model_type,
                    vocab_size=100,
                    num_classes=2,
                    embedding_dim=32,
                    hidden_dim=32,
                    num_filters=16,
                    kernel_sizes=(3, 4, 5),
                )
                model.eval()

                batch = torch.zeros((2, 8), dtype=torch.long)
                with torch.no_grad():
                    logits, weights = model(batch)

                self.assertFalse(torch.isnan(logits).any().item(), f"{model_type} logits contain NaN")
                self.assertFalse(torch.isinf(logits).any().item(), f"{model_type} logits contain Inf")
                self.assertFalse(torch.isnan(weights).any().item(), f"{model_type} weights contain NaN")
                self.assertFalse(torch.isinf(weights).any().item(), f"{model_type} weights contain Inf")

    def test_short_sequences_do_not_crash(self) -> None:
        """Sequences shorter than the largest kernel (e.g. 5) must not crash."""
        for model_type in ["bilstm_attention", "textcnn", "textcnn_max"]:
            for seq_len in [1, 2]:
                with self.subTest(model_type=model_type, seq_len=seq_len):
                    model = build_model(
                        model_type,
                        vocab_size=50,
                        num_classes=2,
                        embedding_dim=16,
                        num_filters=8,
                        kernel_sizes=(3, 4, 5),
                    )
                    model.eval()

                    batch = torch.ones((2, seq_len), dtype=torch.long)
                    with torch.no_grad():
                        logits, weights = model(batch)

                    self.assertEqual(logits.shape, (2, 2))
                    self.assertEqual(weights.shape, (2, seq_len))
                    self.assertFalse(torch.isnan(logits).any().item())
                    self.assertFalse(torch.isnan(weights).any().item())
                    self.assertAlmostEqual(float(weights[0].sum().item()), 1.0, places=3)

    def test_old_bilstm_checkpoint_loads_and_predicts(self) -> None:
        sentiment_ckpt = ROOT / "outputs/artifacts/bilstm_attention.pt"
        if not sentiment_ckpt.exists():
            self.skipTest("Existing bilstm_attention.pt checkpoint not found.")

        classifier = SavedTextClassifier(ROOT / "outputs/artifacts", "bilstm_attention.pt")
        result = classifier.predict("This movie was excellent and enjoyable.")

        self.assertEqual(result["label"], "positive")
        self.assertGreater(result["confidence"], 0.95)
        self.assertGreater(len(result["attention_tokens"]), 0)
        top_token = max(result["attention_tokens"], key=lambda item: item["weight"])["token"]
        self.assertIn(top_token, {"excellent", "enjoyable"})

    def test_old_emotion_checkpoint_loads(self) -> None:
        emotion_ckpt = ROOT / "outputs/emotion/artifacts/emotion_bilstm_attention.pt"
        if not emotion_ckpt.exists():
            self.skipTest("Existing emotion checkpoint not found.")

        classifier = SavedTextClassifier(ROOT / "outputs/emotion/artifacts", "emotion_bilstm_attention.pt")
        result = classifier.predict("This is wonderful news!")
        self.assertIn("label", result)
        self.assertIn("probabilities", result)
        self.assertEqual(len(result["probabilities"]), 7)

    def test_cnn_checkpoint_save_and_xai_explanations(self) -> None:
        """Build, save, load a TextCNN model and verify Attention, LIME, and SHAP work."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            vocab = {"<PAD>": 0, "<UNK>": 1, "great": 2, "terrible": 3, "film": 4, "acting": 5}
            class_names = ["negative", "positive"]

            model_config = {
                "vocab_size": len(vocab),
                "num_classes": 2,
                "embedding_dim": 16,
                "num_filters": 8,
                "kernel_sizes": (3, 4, 5),
                "dropout": 0.1,
                "padding_idx": 0,
                "pooling": "attention",
            }
            model = build_model("textcnn", **model_config)

            checkpoint = {
                "model_state": model.state_dict(),
                "model_type": "textcnn",
                "model_config": model_config,
                "class_names": class_names,
                "max_length": 20,
            }
            ckpt_path = temp_dir / "textcnn.pt"
            torch.save(checkpoint, ckpt_path)
            (temp_dir / "vocabulary.json").write_text(json.dumps(vocab))

            clf = SavedTextClassifier(temp_dir, "textcnn.pt")
            pred = clf.predict("great film")
            self.assertIn(pred["label"], class_names)
            self.assertEqual(len(pred["attention_tokens"]), 2)

            # Attention explanation
            attn_exp = attention_explanation(pred)
            self.assertEqual(attn_exp["method"], "attention")
            self.assertEqual(len(attn_exp["important_words"]), 2)

            # LIME explanation
            lime_exp = lime_explanation(clf, "great film", num_features=2, num_samples=30)
            self.assertEqual(lime_exp["method"], "lime")
            self.assertGreater(len(lime_exp["important_words"]), 0)

            # SHAP explanation
            shap_exp = shap_explanation(clf, "great film", max_tokens=2, nsamples=20)
            self.assertEqual(shap_exp["method"], "shap")
            self.assertGreater(len(shap_exp["important_words"]), 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
