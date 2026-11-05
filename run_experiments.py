"""Train controlled SentixAI model and hyperparameter experiments; save report evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam, AdamW, RMSprop
from torch.utils.data import DataLoader

from comparison_models import MODELS
from sentixai_data import SentimentDataset, build_vocabulary, load_and_split_data


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(name: str) -> torch.device:
    if name in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if name in {"auto", "cpu"}:
        return torch.device("cpu")
    return torch.device(name)


def metrics(y: list[int], pred: list[int]) -> dict[str, float]:
    weighted = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    macro = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(weighted[0]),
        "recall": float(weighted[1]),
        "f1": float(weighted[2]),
        "macro_f1": float(macro[2]),
    }


def epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    ys, ps = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * len(y)
            ys += y.cpu().tolist()
            ps += logits.argmax(1).cpu().tolist()
    return total / len(loader.dataset), ys, ps


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    return {"adam": Adam, "adamw": AdamW, "rmsprop": RMSprop}[name.lower()](
        params, lr=lr, weight_decay=weight_decay
    )


def plot_confusion(y, pred, names, path, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one(name, spec, data, vocab, loaders, class_names, root, seed, device):
    seed_everything(seed)
    out = root / f"{name}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cls = MODELS[spec["model"]]
    keys = ("embedding_dim", "hidden_dim", "dropout", "num_filters", "kernel_sizes", "pooling")
    cfg = {k: spec[k] for k in keys if k in spec}
    if "num_filters" not in cfg and "hidden_dim" in spec:
        cfg["num_filters"] = spec["hidden_dim"]

    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self", "vocab_size", "num_classes"}
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    filtered_cfg = cfg if has_var_kwargs else {k: v for k, v in cfg.items() if k in valid_keys}

    model = cls(len(vocab), len(class_names), padding_idx=vocab["<PAD>"], **filtered_cfg).to(device)
    optimizer = make_optimizer(spec["optimizer"], model.parameters(), spec["learning_rate"], spec["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": [], "val_macro_f1": []}
    best = None
    best_loss = float("inf")
    best_val_macro_f1 = 0.0
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    patience = spec["patience"]
    start = time.perf_counter()

    for idx in range(spec["epochs"]):
        tl, ty, tp = epoch(model, loaders[0], loss_fn, device, optimizer)
        vl, vy, vp = epoch(model, loaders[1], loss_fn, device)
        vm = metrics(vy, vp)
        tm = metrics(ty, tp)
        history["train_loss"].append(tl)
        history["validation_loss"].append(vl)
        history["train_accuracy"].append(tm["accuracy"])
        history["validation_accuracy"].append(vm["accuracy"])
        history["val_macro_f1"].append(vm["macro_f1"])
        print(f"[{name} | seed {seed}] epoch {idx + 1}/{spec['epochs']} | "
              f"train loss {tl:.4f}, acc {tm['accuracy']:.4f} | "
              f"val loss {vl:.4f}, acc {vm['accuracy']:.4f}, macro-F1 {vm['macro_f1']:.4f}")
        if vl < best_loss:
            best_loss = vl
            best = copy.deepcopy(model.state_dict())
            best_val_loss = vl
            best_val_macro_f1 = vm["macro_f1"]
            best_val_accuracy = vm["accuracy"]
            patience = spec["patience"]
        else:
            patience -= 1
            if patience == 0:
                print(f"[{name} | seed {seed}]: early stopping triggered")
                break

    elapsed = time.perf_counter() - start
    model.load_state_dict(best)
    test_loss, y, pred = epoch(model, loaders[2], loss_fn, device)
    m = metrics(y, pred)
    report = classification_report(y, pred, target_names=class_names, zero_division=0, output_dict=True)
    params = sum(p.numel() for p in model.parameters())
    checkpoint = out / "model.pt"
    torch.save({"model": spec["model"], "config": cfg, "state_dict": model.state_dict(), "class_names": class_names}, checkpoint)
    size = checkpoint.stat().st_size
    plot_confusion(y, pred, class_names, out / "confusion_matrix.png", f"{name} (seed {seed})")

    rec_keys = (
        "learning_rate", "batch_size", "epochs", "dropout", "hidden_dim",
        "embedding_dim", "optimizer", "weight_decay", "num_filters", "kernel_sizes", "pooling"
    )
    record = {
        "name": name,
        "seed": seed,
        "model": spec["model"],
        **{k: spec[k] for k in rec_keys if k in spec},
        "val_loss": best_val_loss,
        "val_accuracy": best_val_accuracy,
        "val_macro_f1": best_val_macro_f1,
        "test_loss": test_loss,
        **m,
        "training_seconds": elapsed,
        "parameters": params,
        "model_size_bytes": size,
        "epochs_completed": len(history["train_loss"]),
        "history": history,
        "classification_report": report,
    }
    (out / "result.json").write_text(json.dumps(record, indent=2))
    return record


def plot_summary(records, root, filename, title):
    labels = [r["name"] for r in records]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.7), 5))
    x = np.arange(len(labels))
    accs = [r["accuracy"] for r in records]
    acc_errs = [r.get("accuracy_std", 0.0) for r in records]
    f1s = [r["macro_f1"] for r in records]
    f1_errs = [r.get("macro_f1_std", 0.0) for r in records]
    ax.bar(x - .18, accs, .36, yerr=acc_errs, capsize=3, label="Test Accuracy (mean ± std)", color="#3b82f6")
    ax.bar(x + .18, f1s, .36, yerr=f1_errs, capsize=3, label="Test Macro-F1 (mean ± std)", color="#10b981")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(root / filename, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_config.json")
    parser.add_argument("--csv", help="Override the dataset path in the config.")
    parser.add_argument("--output-dir", help="Override the output directory in the config.")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], help="Override the compute device in the config.")
    parser.add_argument("--seeds", help="Comma-separated seeds, e.g. '42,43,44'")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config["dataset"] = args.csv or config["dataset"]
    config["output_dir"] = args.output_dir or config["output_dir"]
    config["device"] = args.device or config["device"]
    dataset_path = Path(config["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Pass --csv with the IMDB CSV path.")

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif "seeds" in config:
        seeds = list(config["seeds"])
    else:
        seeds = [config.get("seed", 42)]

    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    shared = config["shared"]
    device = select_device(config["device"])

    print(f"Starting experiments across {len(seeds)} seed(s): {seeds}")
    print(f"Device: {device}")
    total_start = time.perf_counter()

    all_seed_records = []
    runs_template = []
    for model in config["model_comparison"]:
        runs_template.append((f"comparison_{model}", {"model": model}))
    for experiment in config.get("hyperparameter_experiments", []):
        runs_template.append((experiment["name"], experiment))

    for seed in seeds:
        print(f"\n{'=' * 30} SEED {seed} {'=' * 30}")
        data = load_and_split_data(config["dataset"], seed)
        vocab = build_vocabulary(data.train_tokens, shared["min_frequency"], shared["max_vocab_size"])
        datasets = [
            SentimentDataset(t, l, vocab, shared["max_length"])
            for t, l in (
                (data.train_tokens, data.train_labels),
                (data.validation_tokens, data.validation_labels),
                (data.test_tokens, data.test_labels),
            )
        ]
        print(f"Split sizes (train/val/test): {[len(x) for x in datasets]}; Vocab size: {len(vocab)}")

        for name, override in runs_template:
            spec = {**shared, **override}
            gen = torch.Generator().manual_seed(seed)
            loaders = (
                DataLoader(datasets[0], batch_size=spec["batch_size"], shuffle=True, generator=gen),
                DataLoader(datasets[1], batch_size=spec["batch_size"]),
                DataLoader(datasets[2], batch_size=spec["batch_size"]),
            )
            record = train_one(name, spec, data, vocab, loaders, data.class_names, root, seed, device)
            all_seed_records.append(record)

    # Save all raw runs to seed_runs.csv
    raw_fields = [
        "name", "seed", "model",
        "val_macro_f1", "val_accuracy", "val_loss",
        "macro_f1", "accuracy", "precision", "recall", "f1", "test_loss",
        "training_seconds", "parameters", "model_size_bytes",
        "learning_rate", "batch_size", "epochs", "dropout",
        "hidden_dim", "embedding_dim", "num_filters", "kernel_sizes", "pooling", "optimizer"
    ]
    with (root / "seed_runs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=raw_fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in raw_fields} for r in all_seed_records])

    # Aggregate mean and std across seeds for each unique experiment name
    metric_keys = [
        "val_macro_f1", "val_accuracy", "val_loss",
        "macro_f1", "accuracy", "precision", "recall", "f1", "test_loss",
        "training_seconds"
    ]
    rec_keys = (
        "learning_rate", "batch_size", "epochs", "dropout", "hidden_dim",
        "embedding_dim", "optimizer", "weight_decay", "num_filters", "kernel_sizes", "pooling"
    )

    summary_records = []
    unique_names = [name for name, _ in runs_template]
    for name in unique_names:
        matching = [r for r in all_seed_records if r["name"] == name]
        first = matching[0]
        agg = {
            "name": name,
            "model": first["model"],
            **{k: first[k] for k in rec_keys if k in first},
            "parameters": first["parameters"],
            "model_size_bytes": first["model_size_bytes"],
            "seeds_evaluated": len(matching),
        }
        for m_key in metric_keys:
            vals = [r[m_key] for r in matching]
            agg[m_key] = float(np.mean(vals))
            agg[f"{m_key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_records.append(agg)

    # Select best model by validation macro-F1 mean
    best = max(summary_records, key=lambda r: r["val_macro_f1"])

    summary_fields = [
        "name", "model",
        "val_macro_f1", "val_macro_f1_std",
        "val_accuracy", "val_accuracy_std",
        "val_loss", "val_loss_std",
        "macro_f1", "macro_f1_std",
        "accuracy", "accuracy_std",
        "precision", "precision_std",
        "recall", "recall_std",
        "f1", "f1_std",
        "test_loss", "test_loss_std",
        "training_seconds", "training_seconds_std",
        "parameters", "model_size_bytes",
        "learning_rate", "batch_size", "epochs", "dropout",
        "hidden_dim", "embedding_dim", "num_filters", "kernel_sizes", "pooling", "optimizer"
    ]
    with (root / "comparison_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=summary_fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in summary_fields} for r in summary_records])

    total_elapsed = time.perf_counter() - total_start
    summary_data = {
        "selection_metric": "val_macro_f1",
        "best_experiment": best,
        "summary_results": summary_records,
        "seeds": seeds,
        "total_runtime_seconds": total_elapsed,
        "device": str(device),
    }
    (root / "experiment_summary.json").write_text(json.dumps(summary_data, indent=2))

    # Regenerate charts
    plot_summary([r for r in summary_records if r["name"].startswith("comparison_")], root, "model_comparison_chart.png", "Architecture comparison (Mean ± Std)")
    plot_summary([r for r in summary_records if not r["name"].startswith("comparison_")], root, "hyperparameter_chart.png", "Hyperparameter experiments (Mean ± Std)")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Total Runtime: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
    print(f"Device: {device}")
    print(f"Seeds evaluated: {seeds}")
    print("=" * 70)
    print(f"\nBEST MODEL (Selected by Validation Macro-F1: {best['val_macro_f1']:.4f} ± {best.get('val_macro_f1_std', 0.0):.4f}):")
    print(f"Model Name:  {best['name']} ({best['model']})")
    print("\n--- Validation Metrics ---")
    print(f"  Val Loss:      {best['val_loss']:.4f} ± {best.get('val_loss_std', 0.0):.4f}")
    print(f"  Val Accuracy:  {best['val_accuracy']:.4f} ± {best.get('val_accuracy_std', 0.0):.4f}")
    print(f"  Val Macro-F1:  {best['val_macro_f1']:.4f} ± {best.get('val_macro_f1_std', 0.0):.4f}")
    print("\n--- Test Metrics (Held-out Set) ---")
    print(f"  Test Loss:     {best['test_loss']:.4f} ± {best.get('test_loss_std', 0.0):.4f}")
    print(f"  Test Accuracy: {best['accuracy']:.4f} ± {best.get('accuracy_std', 0.0):.4f}")
    print(f"  Test Macro-F1: {best['macro_f1']:.4f} ± {best.get('macro_f1_std', 0.0):.4f}")
    print(f"  Test F1 (w):   {best['f1']:.4f} ± {best.get('f1_std', 0.0):.4f}")
    print(f"  Test Precision:{best['precision']:.4f} ± {best.get('precision_std', 0.0):.4f}")
    print(f"  Test Recall:   {best['recall']:.4f} ± {best.get('recall_std', 0.0):.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
