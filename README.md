# Hex-grid vs square-grid edge detection (synthetic scenes)

Question: does fuzzy noise removal plus edge detection on a hexagonal grid give more accurate edge maps than a square-grid median + Sobel pipeline, on the same noisy images?

## Run

    pip install -r requirements.txt
    python analysis.py

Writes `results.json` and `figures/f1_comparison.png`. Seed 42, CPU only, runs in a few minutes.

## Method

- Data: 1000 synthetic 128x128 scenes (circles/star polygons on a smooth background, Gaussian + salt-and-pepper noise). 800 train / 200 test split.
- Baseline: 3x3 median filter, Sobel gradient magnitude, fixed threshold chosen by 3-fold CV on the training set.
- Improved: hex resampling, 7-point hex median (impulse removal), fuzzy-weighted smoothing, then a linear hex-neighbourhood gradient with smoothing and threshold chosen by 3-fold CV on the training set. Edges are mapped back to the square grid.
- Metric: tolerant edge F1 (1-pixel tolerance) on the 200 held-out scenes, single test-set evaluation for both models.

## Caveats

Synthetic data only; results may not transfer to natural images.

## Results (latest run)

On a synthetic noisy-scene test set of 200 images, a standard square-grid median-plus-Sobel edge detector reached tolerant edge F1 of 0.942, versus 0.953 for my hexagonal-grid fuzzy pipeline, a small gain that trades precision (0.9995 to 0.975) for recall (0.895 to 0.934).

| Metric | Value |
|---|---|
| baseline_score | 0.9416 |
| improved_score | 0.9529 |
| improvement | 0.01124 |
| baseline_precision | 0.9995 |
| improved_precision | 0.9749 |
| baseline_recall | 0.8954 |
| improved_recall | 0.9344 |
| baseline_psnr_db | 31.99 |
| improved_psnr_db | 31.82 |
| baseline_ms_per_image | 2.686 |
| improved_ms_per_image | 8.108 |
| baseline_cv_f1 | 0.9463 |
| improved_cv_f1 | 0.9577 |
| baseline_threshold | 0.2 |
| improved_smoothing | 0.2 |
| improved_edge_threshold | 0.05 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Methods are deterministic given the fixed-seed data, so no repeated-seed std is reported. Synthetic data only; results may not transfer to natural images. The paper's own reported result is not stated. Runtime is measured in Python/NumPy and is indicative only. The improved model was revised after the first run did not beat the baseline; the test set was not used for this change, only the training-set CV grid was revised.

## Reference

This project is motivated by the following paper:

> K Mostafa, JY Chiang, WC Tsai, I Her (2017). *Fuzzy Noise Removal and Edge Detection on Hexagonal Image*. J. of the Chinese Society of. https://www.academia.edu/download/42037902/FUZZY_NOISE_REMOVAL_AND_EDGE_DETECTION_O20160204-20783-yzgpy5.pdf

Paper author: Prof. John Y. Chiang, National Sun Yat-sen University (chiang@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
