# Learning from Sound: ESC-50 environmental sound classification

Final project for Introduction to Deep Learning, Fall 2026.

**Question:** How close does a small CNN trained from scratch on log-mel spectrograms get to
a large pretrained audio model (CLAP), what does each cost in size and speed, and do they hold
up on sounds recorded with our own microphone?

## Data
ESC-50 (Piczak, 2015): 2000 five-second clips, 50 classes, 40 clips per class,
5 predefined folds. Source: https://github.com/karolpiczak/ESC-50.
Licence: dataset CC BY-NC 3.0; individual clips come from Freesound under their own CC licences.

**Splits:** folds 1-3 train, fold 4 validation, fold 5 test. The folds keep clips from the same
source recording together, so there is no leakage. All choices (C, augmentation, epochs) are made
on validation; the test fold is only evaluated with `--final`.

The ESC-50 files are not in this repository; `prepare_data.py` downloads them.

## Setup
```bash
pip install -r requirements.txt
```
All commands are run from the repository root.

## Reproduce
```bash
# 1. data + baselines
python src/prepare_data.py              # downloads ESC-50 (~600 MB), caches 16 kHz MFCC/log-mel features
python src/prepare_data.py --limit 200  # (optional) quick pipeline test on a subset
python src/baselines.py                 # majority + MFCC/LogReg -> validation

# 2. CNN from scratch
python src/prepare_cnn_features.py      # caches 22.05 kHz, 128-band log-mels for the CNN
python src/train_cnn.py --augment none          --seed 0   # ~7.5 min each on an RTX 3060 laptop GPU
python src/train_cnn.py --augment specaug       --seed 0
python src/train_cnn.py --augment specaug_mixup --seed 0   # repeat with --seed 1, 2

# 3. pretrained CLAP (frozen)
python src/extract_embeddings.py        # downloads laion/clap-htsat-unfused, caches embeddings (once)
python src/train_probe.py               # zero-shot, linear probe, MLP head -> validation

# 4. final numbers (once, at the end)
python src/train_probe.py --final       # CLAP test rows
python src/evaluate.py                  # baselines + saved CNN checkpoints on the test fold, error analysis
python src/make_figures.py              # report figures -> runs/figures/

# 5. real-world test on our own recordings (see "Own recordings" below)
python src/eval_recordings.py
```
`evaluate.py` scores the **saved** best-validation CNN checkpoints, so the test fold never influences
training or model choice.
Every run is appended to `runs/results.csv`. CNN learning curves go to `runs/curves/`, and
checkpoints go to `models/`. Only `models/cnn_specaug_mixup_s1.pt`, the one the demo uses, is committed.

## Preprocessing
Each model has one preprocessing function, which training and the demo both use:
- **MFCC baseline** (`src/audio.py`): mono, 16 kHz, 5 s, peak-normalised; MFCC mean+std.
- **CNN** (`src/cnn_features.py`): mono, 22.05 kHz, 5 s (centre crop), peak-normalised; 128-band log-mel (128 x 216).
- **CLAP** (`preprocess_for_clap` in `src/embed.py`): mono, 48 kHz, 5 s, peak-normalised.

## Results

Model choices were made on validation (fold 4). Only the chosen variants were scored on test (fold 5).

| Model | Val acc | Val macro-F1 | Test acc | Test macro-F1 |
|---|---|---|---|---|
| Majority class | 0.020 | 0.001 | 0.020 | 0.001 |
| MFCC mean+std + logistic regression (C=0.1) | 0.450 | 0.426 | 0.412 | 0.390 |
| CNN from scratch, no augmentation (3 seeds) | 0.829 ± 0.010 | 0.822 ± 0.007 | – | – |
| CNN + SpecAugment (3 seeds) | 0.838 ± 0.003 | 0.831 ± 0.004 | – | – |
| CNN + SpecAugment + mixup (3 seeds) | **0.863 ± 0.007** | **0.862 ± 0.007** | **0.802 ± 0.004** | **0.788 ± 0.008** |
| CLAP zero-shot, prompt "the sound of a X" | 0.860 | 0.834 | – | – |
| CLAP zero-shot, prompt "This is a sound of X." | 0.903 | 0.897 | 0.893 | 0.886 |
| CLAP linear probe (C=10) | **0.983** | **0.983** | **0.963** | **0.961** |
| CLAP MLP head (3 seeds) | 0.976 ± 0.001 | 0.976 ± 0.001 | 0.961 ± 0.001 | 0.959 ± 0.002 |

± is the standard deviation over seeds 0, 1 and 2.
- **CNN:** 1.19M parameters, ~7.5 min training, 25-35 ms CPU inference per clip.
- **Augmentation:** mixup gives +3.4 points over no augmentation (all 3 mixup seeds beat all 3 no-augmentation seeds). SpecAugment alone is within noise.
- **Zero-shot prompt:** the wording changes accuracy by 4 points.

- **Validation → test drop:** every model scores lower on test (CNN −6.1 points, MFCC −3.8, CLAP probe −2.0).
  Fold 5 seems harder in general. The CNN's validation score is also slightly optimistic, because
  validation picked its epoch and augmentation.

**Caveat:** CLAP's training data (LAION-Audio-630K) includes Freesound audio, and ESC-50 is cut from
Freesound. Some ESC-50 clips may have been seen during CLAP pretraining, so the CLAP numbers may be optimistic.

## Error analysis (test fold)
Full details are in `runs/error_analysis.md`, with per-clip predictions in `runs/test_predictions.csv`.
Figures are in `runs/figures/`.
- **The CNN struggles most with steady, noisy sounds that have little structure:** airplane (1/8 correct),
  helicopter, wind, engine and washing machine. These get confused with one another and with
  thunderstorm and siren. The exterior/urban group is its weakest (0.70, vs 0.81-0.85 for the other groups).
- **Similar textures get mixed up:** fireworks → crackling fire, clapping → rain (dense clicks),
  and hen → laughing or crying baby (voice-like pitch).
- **CLAP fixes most of these** (15 errors vs 77). Its remaining errors are mostly helicopter
  (→ airplane, washing machine) and laughing → crying baby.
- **CLAP zero-shot fails on vague label names:** insects (1/8), wind (2/8), and hen → rooster (5 times).
  This shows the text prompt, not the audio, is the weak point there.

## Own recordings (real-world test)
The easiest way to record is in the app's **Classify** tab: record with the microphone, open
**"Save this clip to my test set"**, pick the true class and click **Save to test set**. That saves
`data/my_recordings/<class>_<NN>.wav`, using the same microphone and pipeline as the live demo.
Recordings from other apps (a phone's voice recorder, Windows Sound Recorder) also work: copy them
into `data/my_recordings/`, named by ESC-50 class, e.g. `door_wood_knock_01.wav`
or `keyboard_typing_2.m4a`. This folder is a **test set and is never trained on**. The app's
*Add training clips* tab saves to `data/user_samples/` instead. Score the recordings in the app's
*My recordings* tab, or with `python src/eval_recordings.py`.

## Sound Lab app (live demo)
```bash
python -m streamlit run app.py          # opens http://localhost:8501
python src/inference.py --selftest      # quick check of all three models on held-out test-fold clips
```
One app with four tabs:
- **🎙️ Classify:** record from the microphone, upload a clip, or pick a held-out test clip. Shows the
  top-5 of the CNN, CLAP probe and CLAP zero-shot side by side. Recordings can be saved to the test set.
- **➕ Add training clips:** add labelled clips (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) to
  `data/user_samples/` (committed). They become extra training data for the CLAP probe.
- **📊 Train & compare:** retrain the CLAP probe with and without the added clips. It is evaluated on
  ESC-50 fold 4 and on our own recordings. It does not change the official folds, caches or test-fold scores.
- **🧪 My recordings:** listen to and delete test recordings, and score all three models on them,
  compared with the official test fold.

## Layout

| File | Purpose |
|---|---|
| `src/audio.py` | 16 kHz preprocessing + MFCC features (baseline) |
| `src/prepare_data.py` | Download ESC-50, cache baseline features |
| `src/baselines.py` | Majority + MFCC/LogReg; `log()` writes `runs/results.csv` |
| `src/cnn_features.py` | CNN preprocessing (22.05 kHz log-mel) |
| `src/prepare_cnn_features.py` | Cache CNN log-mels |
| `src/train_cnn.py` | Small 2D CNN, SpecAugment / mixup, early stopping on validation |
| `src/embed.py` | CLAP loading, preprocessing and embedding helpers |
| `src/extract_embeddings.py` | Cache CLAP audio + text embeddings |
| `src/train_probe.py` | CLAP zero-shot, linear probe, MLP head |
| `src/evaluate.py` | Final test-fold evaluation and error analysis |
| `src/make_figures.py` | Report figures |
| `src/eval_recordings.py` | Score all models on our own recordings |
| `src/inference.py` | Loads the three trained models once and classifies a clip (shared by the app and scripts) |
| `app.py` | Sound Lab: the Streamlit app (live demo, training clips, comparison, own recordings) |
