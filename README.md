# Music Diffusion + LSTM Music Generation (Maqam / Oud + MIDI Experiments)

This repository contains a set of **music generation experiments** built across the semester, covering two main directions:

1) **Audio diffusion** (DDPM-style denoising with a U-Net backbone) and supporting utilities for forward noising and sampling.  
2) **Token-based music generation with LSTM**, producing note sequences and exporting them to MIDI.

The included report (PDF/LaTeX in your docs) is meant as a **guide/explanation**, but the repo itself is primarily a **code + experiment workspace**.

---

## What this repo does

### 1) Diffusion-based generation (audio/spectrogram side)
- Implements **forward noising** and a **denoiser model (U-Net)** for diffusion-style training.
- Provides scripts to experiment with diffusion pipelines and run training/sampling loops.

Relevant files:
- `U_net_denoiser.py` — U-Net denoiser architecture for diffusion
- `forward_noising.py` — forward diffusion/noising utilities
- `music_diffusion.py` — main diffusion pipeline (training/sampling orchestration)

---

### 2) LSTM music generator (token/MIDI side)
- Trains an **LSTM** on tokenized music data (from `train.json`) and generates new sequences.
- Exports generated tokens to:
  - `generated_notes.json` (tokens/notes)
  - `generated_notes.mid` (MIDI output)
- Includes iterative improvements of data loading + training behavior over multiple versions.

Relevant files:
- `LSTM.py` — training + generation entry-point for LSTM experiments
- `Model_LSTM.py` — LSTM model definition
- `helper_LSTM.py` — helper functions (token handling, sampling, MIDI utilities)
- `Conditional_data_Loading.py` — dataset/data loading logic
- `Lstm_data_visulatization.ipynb` — visualization/debug notebook for data/training
- `lstm_music_model.pth` — saved checkpoint (one of the working versions)

---

## Repo contents (high level)

- **Diffusion**
  - `music_diffusion.py`
  - `U_net_denoiser.py`
  - `forward_noising.py`
  - `Autoencoder.py` (utilities for latent compression experiments)

- **LSTM**
  - `LSTM.py`
  - `Model_LSTM.py`
  - `helper_LSTM.py`
  - `Conditional_data_Loading.py`
  - `train.json`
  - `results_lstm/`, `final_version/` (experiment outputs/iterations)

- **Generated outputs / test files**
  - `generated_notes.json`
  - `generated_notes.mid`
  - `Bach_test.mid`
  - `test_fixed_seed_changed_fs_1000.mid`

- **Utilities**
  - `gpu_checker.py`
  - `test.py`

---

## Quick start

### Setup
Create a virtual environment and install requirements (edit as needed for your machine):
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch numpy matplotlib
