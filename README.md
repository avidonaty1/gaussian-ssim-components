# Gaussian-Weighted SSIM (RGB)

Custom implementation of the Structural Similarity Index (SSIM) for RGB images using Gaussian-weighted local statistics and optional foreground masking.
Allows for component separation of Luminance, Contrast, and Structure as well as overall SSIM.

## Overview

SSIM is computed **per pixel (element-wise)** using Gaussian-windowed local statistics, producing a full similarity map.  
The final scalar score is obtained by averaging the per-pixel SSIM values.

For each pixel location:

luminance(x, y) = 
((2μ_xμ_y + C1) + C1) / 
(μ_x² + μ_y² + C1)

contrast(x, y) = 
((2σ_xσ_y) + C2) /
(σ_x² + σ_y² + C2)

structure(x, y) = 
(σ_xy + C3) /
(σ_xσ_y + C3)

SSIM(x, y) = 
luminance(x, y) * contrast(x, y) * structure(x, y)

Where:

- μx, μy = Gaussian-weighted local means  
- σx², σy² = Gaussian-weighted local variances  
- σxy = Gaussian-weighted local covariance  
- C1, C2 = stability constants  

The similarity map is computed independently per RGB channel and then averaged across channels.

---

## Implementation Decisions

### Gaussian Window (11×11, σ = 1.5)

- Matches widely used reference implementations.
- Gaussian weighting emphasizes central pixels, reducing edge artifacts compared to uniform filtering.
- Provides a balance between spatial locality and statistical stability.

### Element-wise Formulation

- Components of SSIM are computed at each spatial location using local statistics.
- Produces a full-resolution similarity map, as well as maps for Luminance, Contrast, and Structure.
- Final score = mean of the map (after optional masking).

### Stability Constants (C1, C2)

- Chosen based on literature conventions (not dynamically derived from `data_range`).
- Designed to prevent instability in low-variance or low-luminance regions.

### Foreground Masking

Optional foreground masking allows evaluation only over relevant regions.

Masking works by:

1. Generating or supplying a binary foreground mask (this is done in LAB color space).
    - To get the background color value, only A and B of LAB is used. This means that the background is Luminance-invariant, allowing for masking out a background that is unevenly lit or partially in shadow. Any pixel index where the magnitude of 
    [A B] is different enough than that of the background value (tunable parameter) is assigned to the foreground.
2. Computing the full Luminance, Contrast, Structure, and SSIM map.
3. Averaging only the values where the mask == 1.

This prevents large uniform background areas from artificially inflating similarity scores and is useful when comparing segmented objects or reconstructed subjects.

---

## Features

- Configurable Gaussian window size
- Gaussian-weighted local statistics
- Per-channel RGB SSIM aggregation
- Full similarity map output for individual components
- Optional foreground masking

## Usage

```bash
pip install -r requirements.txt
python ssim.py {image_1} {image_2}
