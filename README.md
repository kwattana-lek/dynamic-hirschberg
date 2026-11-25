# Dynamic Hirschberg (DH)

Code and data for the manuscript:

> Suwannarat K., Kurdthongmee W., Phetkaew T., Wattanapanich C., Abd Aziz A.Z., Kurdthongmee P.
> *Dynamic Hirschberg: Calibration-Free Camera-Based Temporal Sensing of Ocular Alignment Using a CNN–BiLSTM Framework.* Submitted to *Sensors* (MDPI), 2026.

DH tracks the distance between the pupil center and the corneal reflex (glint) of both eyes over a 5-s video. It divides that distance by the iris diameter, so no per-user calibration is needed, and classifies the 150-frame trajectory as **Healthy**, **Intermittent** or **Constant** deviation with a compact CNN–BiLSTM.

**This is a research prototype. It is not a clinically validated diagnostic tool.**

## Repository contents

```
data/synthetic_dataset_robust/      600 synthetic sequences (200 per class); columns L_ratio, R_ratio
src/
  generate_synthetic_dataset.py     synthetic temporal sequence generator (STSG) that produced data/
  dh_video_inference.py             video pipeline: MediaPipe Face Mesh -> pupil/glint -> ratio -> classifier
  train_model_original.py           original training script (kept for transparency; not used for reported results)
experiments/
  rerun_experiments.py              ALL synthetic experiments in the paper (Tables 2-6, Figures 4-5)
  run_real_videos.py                real-video feasibility check (Table 7)
  glint_diag.py                     peak grey level near the iris (Table 7)
  plot_fig2.py, plot_fig3.py        Figures 2 and 3
  plot_pooled_cm.py                 pooled confusion matrix (Figure 4) and Table 3
  results.json                      every number reported in Tables 2-6
  pooled_cnn_bilstm.json            Table 3
  predictions/                      saved test-set class probabilities for every model and seed
  figures/                          figures as used in the paper
  real_video/                       per-frame signals and per-window predictions for the 13 videos, the
                                    Dreamstime clip IDs, and the seed-0 CNN–BiLSTM weights (cnn_bilstm_seed0.keras)
```

## Installation

Python 3.11 is recommended (tested on macOS, Apple M4).

```bash
python -m venv dh_env
source dh_env/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
cd experiments
python rerun_experiments.py      # ~10 min on a laptop CPU; writes results.json, predictions/, figures/
python plot_pooled_cm.py         # Table 3 and Figure 4
python plot_fig2.py              # Figure 2
python plot_fig3.py              # Figure 3
```

The split is stratified 70/15/15 (seed 42), so the test set has 90 sequences. Every network is trained with seeds 0–4, using Adam (learning rate 1e-3), batch size 32 and early stopping on validation loss (patience 10). The test set is never used for training or model selection.

### Real-video check

The 13 videos are Dreamstime stock clips licensed under the Dreamstime Standard Content License. They are **not** redistributed here. Their clip IDs are in `experiments/real_video/dreamstime_video_ids.csv`. Put the purchased clips in `Normal/` and `Strabismus/` sub-folders, then run:

```bash
export DH_VIDEO_DIR=/path/to/videos
python run_real_videos.py
python glint_diag.py
```

Only per-frame numerical values are saved. No images are written.

## Measurement pipeline (summary)

For each frame, MediaPipe Face Mesh is run with `refine_landmarks=True`.
- The pupil center is iris landmark 468 for one eye and 473 for the other.
- The iris diameter is the distance between landmarks 469 and 471 (and 474 and 476 for the other eye).
- The glint is the brightest pixel of a 3×3-Gaussian-blurred grey patch within ±5 px of the pupil center, accepted only if its value is at least 200.
- The ratio is |pupil − glint| / iris diameter. It is set to 0 when no face or no glint is found.

**Requirements and known limitations (see the paper):**
- A visible corneal reflex is required, so a light source near the camera must be used.
- An all-zero (failed) sequence is currently classified as Healthy. A quality gate is needed before any real use.
- The pipeline outputs an unsigned distance, whereas the generator produces signed values.

## Ethics

This work is part of the project "Intelligent System for Gaze Analysis and Ocular Abnormality Screening Using Deep Learning Techniques on Secondary Datasets". The project received a certificate of exemption from the Human Research Ethics Committee of Walailak University (project no. WU-EC-EN-4-461-68; certificate WUEC-26-020-01; approved 15 January 2026).

## Citation

Please cite the paper above. The citation will be updated after publication.

## Contact

Wattanapong Kurdthongmee, School of Engineering and Technology, Walailak University, Thailand.

## License

Code: MIT License (see `LICENSE`). The Dreamstime videos are not part of this repository and remain under the Dreamstime licence.
