# Fuzzy Edge Detection on a Hexagonal Grid vs a Square Grid

Motivation: Fuzzy noise removal and edge detection on hexagonal images, as proposed in "Fuzzy Noise Removal and Edge Detection on Hexagonal Image" (2017), suggests that hexagonal sampling, where each pixel has six equidistant neighbours, may reduce directional bias in edge detection. This experiment checks that claim under controlled conditions against a standard square-grid pipeline.

Question: Does fuzzy noise removal and edge detection on a hexagonal pixel grid produce more accurate edge maps than the same pipeline on a standard square grid, using the same noisy images?

Dataset: synthetic 128x128 grayscale scenes (random circles and polygons on a smooth background) with exact ground-truth edge maps, Gaussian plus salt-and-pepper noise, generated in code with a fixed seed. No download is needed. This is not the paper's own dataset. The test split is 20% of the generated images, and the same noisy images are given to both models.

Baseline: a square-grid pipeline with a 3x3 median filter followed by Sobel gradient magnitude and a fixed threshold. The threshold is chosen by cross-validation on the training split.

Improved model: the image is resampled onto an offset-row hexagonal grid. Fuzzy-membership smoothing and a fuzzy edge rule are applied over the six hex neighbours, and the edges are mapped back to the square grid for scoring. The smoothing width and edge threshold are chosen by cross-validation on the training split.

How to run:

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and a chart to figures/f1_comparison.png).

## Reference

This project is motivated by the following paper:

> K Mostafa, JY Chiang, WC Tsai, I Her (2017). *Fuzzy Noise Removal and Edge Detection on Hexagonal Image*. J. of the Chinese Society of. https://www.academia.edu/download/42037902/FUZZY_NOISE_REMOVAL_AND_EDGE_DETECTION_O20160204-20783-yzgpy5.pdf

Paper author: Prof. John Y. Chiang, National Sun Yat-sen University (chiang@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
