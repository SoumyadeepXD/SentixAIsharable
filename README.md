# SentixAI Analytics

**Explainable Deep Learning for Sentiment and Emotion Analysis**

**Live demo:** [https://sentixai-01cn.onrender.com](https://sentixai-01cn.onrender.com)

## Abstract

SentixAI is a local web application for English sentiment and emotion analysis. It uses independently trained BiLSTM-attention models, shows confidence and class probabilities, and offers attention, LIME, and SHAP explanations. It also supports batch CSV processing and a reproducible neural-model comparison study.

## Problem statement and objectives

Text classifiers often return a label without explaining it. SentixAI predicts sentiment/emotion, surfaces model evidence, supports batch analysis, and preserves metrics/experiments for a B.Tech CSE-AIML demonstration.

## Features

- Saved-model inference: the dashboard never retrains a model.
- Single-review and CSV batch analysis.
- Confidence scores, real model probabilities, and downloadable batch results.
- Attention, optional LIME, and optional SHAP explanations.
- Training curves, real confusion matrices, and saved evaluation reports.
- MLP, LSTM, BiLSTM, and BiLSTM-attention comparison runner.

## System architecture

```text
Browser dashboard / CSV upload
        ↓
FastAPI (app.py)
        ↓
Shared preprocessing + saved vocabulary
        ↓
Sentiment checkpoint ─→ label, probability, explanation
Emotion checkpoint   ─→ label, probability, explanation
        ↓
Dashboard cards, charts, and downloadable CSV
```

## Dataset

### Sentiment

The IMDB CSV uses `review` and `sentiment` columns. Duplicate/empty rows are removed; the recorded run contained 49,582 usable reviews. Its genuine labels are `positive` and `negative` only.

### Emotion

The emotion model uses filtered public GoEmotions data. Only exact, single-label source classes are retained: `anger`, `disappointment`, `fear`, `joy`, `love`, `sadness`, and `surprise`.

## Data preprocessing

Text is lowercased, HTML entities/tags and URLs are removed, and word-like English tokens are extracted. A vocabulary with `<PAD>`/`<UNK>` is fitted only on training text. Sequences are padded or truncated to a fixed length.

## Neural network architecture

```text
Tokens → Embedding → BiLSTM → Additive Attention → Dropout → Dense logits → Softmax
```

`CrossEntropyLoss` receives logits while softmax is used only to produce inference probabilities.

## BiLSTM and attention mechanism

The BiLSTM reads token sequences in both directions. Additive attention gives each non-padding token a normalized weight and pools the BiLSTM states into a context vector. These actual weights are exposed in inference output.

## TextCNN architecture and pooling modes

```text
Tokens → Embedding → Parallel Conv1d (length-preserving) → Concat → Pooling (Attention or Max) → Dropout → Dense logits → Softmax
```

TextCNN extracts local n-gram feature maps via parallel 1D convolutions (default kernel sizes 3, 4, and 5 with length-preserving padding) over word embeddings. SentixAI supports two pooling strategies conforming to the `(logits, token_weights)` contract:
- **Additive Attention (`textcnn`):** Reuses parameterized additive attention over the concatenated multi-filter feature sequence, yielding dynamic token weights that sum to 1 over non-padding tokens.
- **Masked Global Max-Pooling (`textcnn_max`):** Performs masked max-pooling across time and returns token weights defined by the proportion of feature channels whose maximal response landed on each token.

### Training TextCNN
```bash
python train_sentixai.py --csv "data/IMDB Dataset.csv" --model-type textcnn --output-dir outputs/cnn_sentiment
```
For max-pooling:
```bash
python train_sentixai.py --csv "data/IMDB Dataset.csv" --model-type textcnn_max --output-dir outputs/cnn_max_sentiment
```

### Architecture Comparison
TextCNN models (`textcnn` and `textcnn_max`) are evaluated side-by-side with MLP, LSTM, and BiLSTM baselines:
```bash
python run_experiments.py --config experiment_config.json --csv "data/IMDB Dataset.csv"
```

## Emotion detection

Emotion is an independently trained model with its own vocabulary, checkpoint, and evaluation report; it is not trained from IMDB sentiment labels.

## Explainable AI: Attention, LIME, and SHAP

- **Attention** displays token weights produced by the model.
- **LIME** perturbs words, runs the saved model, and estimates local word contributions.
- **SHAP** optionally estimates word contributions versus an all-words-masked baseline.

Attention is evidence of model focus, not a causal guarantee. LIME and SHAP are local approximations; Kernel SHAP is intentionally bounded because it is slower.

## Model comparison and hyperparameter experiments

`run_experiments.py` trains MLP, LSTM, BiLSTM, BiLSTM-attention, and TextCNN (attention & max pooling) models on the same stratified split and training vocabulary. `experiment_config.json` controls learning rate, batch size, epochs, dropout, hidden units / filter counts, kernel sizes, embedding dimensions, optimizer, and architecture variants. The best run is selected by **macro-F1**, rather than accuracy alone.

## Evaluation metrics

Each run saves accuracy, precision, recall, weighted F1, macro-F1, classification report, confusion matrix, training curves, training time, parameter count, model size, and checkpoint.

## Results

These are actual held-out test results from completed training runs:

### Production Model Results

| Model | Dataset | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| BiLSTM + Attention sentiment | IMDB | 90.02% | 90.02% | 90.02% | 90.02% |
| BiLSTM + Attention emotion | Filtered GoEmotions | 73.44% | 76.77% | 73.44% | 74.37% |

### Controlled Architecture Comparison Study (IMDB Full Dataset, 3 Seeds)

Evaluated across seeds `[42, 43, 44]` on 49,582 reviews (Train: 39,665, Validation: 4,958, Test: 4,959) using identical splits and vocabulary. Best model selected by validation macro-F1:

| Model Architecture | Val Macro-F1 (Mean ± Std) | Test Macro-F1 (Mean ± Std) | Test Accuracy (Mean ± Std) | Mean Train Time |
|---|---|---|---|---|
| **BiLSTM + Attention** *(Best)* | **0.8795 ± 0.0044** | **0.8810 ± 0.0030** | **88.11% ± 0.31%** | 103.4s |
| BiLSTM | 0.8773 ± 0.0035 | 0.8805 ± 0.0066 | 88.06% ± 0.67% | 109.2s |
| **TextCNN (Attention Pooling)** | 0.8723 ± 0.0012 | 0.8735 ± 0.0055 | 87.37% ± 0.56% | **87.1s** |
| MLP Baseline | 0.8701 ± 0.0058 | 0.8734 ± 0.0076 | 87.34% ± 0.75% | **31.8s** |
| TextCNN (Filters 64, LR 0.0005) | 0.8527 ± 0.0036 | 0.8554 ± 0.0080 | 85.56% ± 0.81% | 83.2s |
| **TextCNN (Global Max-Pooling)** | 0.8448 ± 0.0081 | 0.8478 ± 0.0106 | 84.82% ± 1.02% | 88.3s |
| LSTM Baseline | 0.8133 ± 0.0398 | 0.8161 ± 0.0372 | 81.66% ± 3.75% | 70.2s |

Full raw multi-seed runs are saved in `outputs/experiments/seed_runs.csv` and summary metrics in `outputs/experiments/comparison_results.csv`. Visual comparison charts are in `outputs/experiments/model_comparison_chart.png` and `hyperparameter_chart.png`.

## Dashboard

The FastAPI-served dashboard provides text prediction, batch CSV prediction, probabilities, explanations, confusion matrices, and training/evaluation charts.

## Project structure

```text
app.py                    Dashboard/API and batch-upload routes
static/                   Dashboard HTML, CSS, JavaScript
sentixai_data.py          Cleaning, tokenization, vocabulary, datasets
sentixai_model.py         Production BiLSTM-attention model
sentixai_inference.py     Saved-checkpoint inference used by API/batch jobs
sentixai_xai.py           Attention, LIME, SHAP explanations
train_sentixai.py         Reproducible sentiment/emotion training
prepare_goemotions.py     GoEmotions preparation utility
comparison_models.py      MLP, LSTM, BiLSTM, attention architectures
run_experiments.py        Controlled comparison runner
outputs/                  Ignored generated model/results/download files
data/                     Prepared emotion data and sample batch CSV
```

## Installation

```bash
git clone https://github.com/SoumyadeepXD/SentixAIsharable.git
cd SentixAIsharable
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training

Train sentiment:

```bash
python train_sentixai.py --csv "data/IMDB Dataset.csv" --output-dir outputs
```

Prepare and train emotion:

```bash
python prepare_goemotions.py --output data/goemotions_target_emotions.csv
python train_sentixai.py --csv data/goemotions_target_emotions.csv --text-column text --label-column emotion --output-dir outputs/emotion --model-name emotion_bilstm_attention.pt
```

Run the controlled comparison study:

```bash
python run_experiments.py --config experiment_config.json --csv "data/IMDB Dataset.csv"
```

It writes report-ready results to `outputs/experiments/`, including comparison CSV/JSON files, charts, checkpoints, and per-run confusion matrices.

## Testing

```bash
python predict_sentixai.py --text "This movie was excellent and enjoyable." --model-dir outputs/artifacts
python predict_sentixai_full.py --text "The service was slow and the product arrived damaged."
python explain_sentixai.py --text "The camera is excellent but the battery is terrible." --model sentiment --methods attention lime shap
```

## Running the application

### Option 1: Streamlit Dashboard (Recommended for Live Demo & Sharing)

Run the interactive Explainable AI dashboard with token attention heatmaps, LIME, SHAP, batch CSV analysis, and model curves:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Option 2: FastAPI Backend & Classic Dashboard

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Deployment & Sharable Link (Streamlit Cloud vs Render vs Vercel)

### 1. Deploying to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is 100% free, runs native PyTorch without memory/bundle limits, and provides a permanent public URL (e.g. `https://sentixai.streamlit.app`):

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "Add Streamlit Explainable AI dashboard"
   git push origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New App"** (or **"Create app"**).
4. Select:
   - **Repository:** `SoumyadeepXD/SentixAIsharable`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **"Deploy!"** Your public sharable link will be live in 1-2 minutes!

### 2. Why Not Vercel for PyTorch Deep Learning?

- **Bundle Size Limit:** Vercel Serverless Functions enforce a strict **250MB uncompressed limit**. PyTorch, SciPy, Scikit-learn, LIME, and SHAP together exceed 1.2GB, causing serverless build failure.
- **Execution Timeouts:** XAI computations like LIME and SHAP can take 5-15 seconds, hitting Vercel's strict 10s-15s serverless timeout caps.
- **Vercel Static Hosting:** If desired, Vercel can host the static HTML/CSS/JS frontend (`vercel.json` is provided in the repository) while the models run on a dedicated Python service. However, Streamlit Cloud hosts the full model + UI seamlessly in one place at zero cost.

## Security and reproducibility

The project requires no API keys. `.gitignore` excludes virtual environments, outputs, and `.env` files. Do not commit credentials or private datasets. Training seeds are fixed and deterministic operations are requested where supported by the PyTorch backend.

## Limitations

- Sentiment supports only positive/negative; Neutral is deliberately marked not trained.
- Emotion supports seven source-labelled classes; Frustration is deliberately marked not trained.
- Models are English-focused and reflect IMDB/GoEmotions domains.
- Customer-service complaints can produce unintuitive emotion labels because they are outside the emotion model's main domain.
- LIME/SHAP add inference time; SHAP is bounded.
- Architecture-comparison metrics are pending until the configured study is run.

## Future scope

- Train real three-class sentiment data including neutral.
- Add frustration-labelled customer-support data.
- Add labelled multilingual data and multilingual models.
- Add aspect-based sentiment labels/models.
- Run repeated-seed/cross-validation comparison experiments.
- Add authentication, rate limiting, storage, and monitoring for non-local deployment.
