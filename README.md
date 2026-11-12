# Learning from Sound: ESC-50 environmental sound classification

## Setup
```bash
pip install -r requirements.txt
python src/prepare_data.py            # downloads ESC-50 (~600 MB), caches features
python src/prepare_data.py --limit 200  # quick pipeline test on a subset
python src/baselines.py               # majority + MFCC/LogReg, evaluated on validation
python src/baselines.py --final       # run once at the end: also evaluates the test fold
```
Runs are logged to `runs/results.csv`.

## Data
ESC-50 (Piczak, 2015): 2000 five-second clips, 50 classes, 40 clips per class,
5 predefined folds. Source: https://github.com/karolpiczak/ESC-50.
Licence: dataset CC BY-NC 3.0; individual clips come from Freesound under their own CC licences.
Splits: folds 1-3 train, fold 4 validation, fold 5 test. The folds keep clips from the same
source recording together, so there is no leakage.

## Preprocessing (src/audio.py, shared with the demo)
Mono, 16 kHz, padded/cropped to 5 s, peak-normalised. Features: log-mel (64 bands) and MFCC mean+std.

## Pretrained model: CLAP (frozen) + probes
```bash
python src/extract_embeddings.py        # downloads laion/clap-htsat-unfused, caches embeddings (slow step, once)
python src/train_probe.py               # zero-shot baseline, linear probe, MLP head -> validation
python src/train_probe.py --final       # once at the end: adds test fold
```
CLAP uses its own preprocessing (48 kHz, 5 s, peak-normalised): `preprocess_for_clap` in `src/embed.py`.
