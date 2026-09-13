# Harmonic Perception Trainer

Web ear-training app for practising musical interval and chord recognition. You can use it to generate and listen to exercises, take timed tests, track progress, and recognise intervals from recorded audio via a machine learning model.

The repository also contains the dataset preparation scripts and PyTorch experiments behind the accompanying research paper.

## Features

- **Training** — generate and play intervals and chords
- **Testing** — take timed quizzes, earn XP, and track your progress
- **Recognition** — send recorded audio to the backend for interval classification
- **Experiments** — explore spectrograms and overtone-related visualisations in the app
- **Research** — reproduce the experiments described in the accompanying paper

## Repository layout

```text
backend/                  Django REST API (auth, training audio, tests, recognition)
  core/audio/             Audio generation and processing
  core/recognition/       Audio recognition pipeline
  core/ml/                Model loading and inference
dataset-preparation/      Synthetic and recorded dataset pipelines
experiments/              Research experiments
frontend/                 React + Vite SPA
```

## Tech stack

| Part | Stack |
|------|--------|
| Frontend | React 19, Vite, Tone.js, Axios |
| Backend | Django, Django REST Framework, SimpleJWT |
| Production ML | TensorFlow, librosa, scikit-learn |
| Research ML | PyTorch, torchaudio, librosa, Optuna, Transformers |

## Getting started

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Datasets

Large audio folders are gitignored. To regenerate:

```bash
python dataset-preparation/scripts/generate_synthetic_dataset.py
python dataset-preparation/scripts/process_recorded_dataset.py --positions NF MF
```

### 4. Research experiments

```bash
pip install -r experiments/requirements.txt

python -m experiments.exp1_representations
python -m experiments.exp2_transfer
python -m experiments.exp3_microphone
python -m experiments.exp4_interpretability
```

| Package | Focus |
|---------|--------|
| `exp1_representations` | Mel / CQT / HCQT (and waveform) comparison |
| `exp2_transfer` | Scratch CNNs vs PANNs / AST transfer |
| `exp3_microphone` | Synthetic-to-real microphone transfer |
| `exp4_interpretability` | Grad-CAM, embeddings, FFT–MLP baseline |

Each module accepts CLI flags (representations, seeds, splits, etc.). See the package `config.py` / `__main__.py` for details.

## Publication

The experiments in `experiments/` correspond to the paper:

**Feature engineering and deep learning for musical interval recognition through spectral analysis**  
Discover Artificial Intelligence (2026)  
https://doi.org/10.1007/s44163-026-01499-3
