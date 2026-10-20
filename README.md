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

## Reference

This project is motivated by the following paper:

> K Mostafa, JY Chiang, WC Tsai, I Her (2017). *Fuzzy Noise Removal and Edge Detection on Hexagonal Image*. J. of the Chinese Society of. https://www.academia.edu/download/42037902/FUZZY_NOISE_REMOVAL_AND_EDGE_DETECTION_O20160204-20783-yzgpy5.pdf

Paper author: Prof. John Y. Chiang, National Sun Yat-sen University (chiang@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
