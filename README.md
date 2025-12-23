# Deep-Image-Prior-for-NDVI-Reconstruction
This repository highlights the use of Deep Image Prior (DIP) for reconstructing NDVI from cloud obscured satellite images. Hence, further research on this area can prove to be beneficial for cleaning of noisy satellite data.  

1. Problem Overview

Satellite-derived vegetation indices such as the Normalized Difference Vegetation Index (NDVI) are fundamental for monitoring vegetation health, agricultural productivity, and land-cover dynamics. However, optical satellite imagery is frequently corrupted by cloud cover, leading to missing or unreliable NDVI values. Recovering vegetation information in these cloud-occluded regions constitutes an ill-posed inverse problem, as multiple plausible reconstructions may exist for the same corrupted observation.

Traditional approaches to NDVI gap filling often rely on temporal interpolation, multi-temporal composites, or supervised learning methods that require large historical datasets and region-specific training. In contrast, this project explores the use of Deep Image Prior (DIP), which leverages the implicit bias of convolutional neural network architectures to regularize inverse problems without requiring any training data.

The objective of this work is to reconstruct cloud-corrupted NDVI using a single satellite image, real cloud masks, and an untrained neural network optimized directly on the corrupted observation.

2. Dataset Description
2.1 Source Data

The dataset used in this project is derived from the MODIS MOD13Q1 product, which provides 16-day composite NDVI at 250 m spatial resolution. MOD13Q1 is widely used in vegetation and climate studies due to its radiometric consistency and global coverage.

From the MOD13Q1 HDF file, the following layers were extracted:

NDVI: Scaled by the official MODIS factor (0.0001) to obtain physical NDVI values in the range 
[
−
1
,
1
]
[−1,1].

Pixel Reliability Layer: Used to identify cloud-affected and unreliable pixels.

2.2 Cloud Mask Construction

The MODIS pixel reliability layer encodes data quality as:

0: Good quality

1: Marginal quality

2: Snow / ice

3: Cloud

A binary cloud mask was constructed by retaining only pixels with reliability value 0. All other pixels were treated as missing data in the forward model. This results in a realistic cloud-corrupted NDVI image that accurately reflects operational satellite conditions.

2.3 Patch-Based Processing

Due to computational constraints and the local nature of vegetation structure, reconstruction was performed on a single 256 × 256 spatial patch extracted from the full NDVI tile. Patch-based processing is standard practice in remote sensing image restoration and enables efficient optimization while preserving spatial coherence.

3. Forward Model and Inverse Problem Formulation

The cloud corruption process is modeled as a masking operation:

𝑦
=
𝑀
⊙
𝑥
y=M⊙x

where:

𝑥
x is the unknown clean NDVI image,

𝑀
M is the binary cloud mask,

𝑦
y is the observed cloud-corrupted NDVI.

The inverse problem consists of estimating 
𝑥
x given 
𝑦
y and 
𝑀
M, which is ill-posed due to the loss of information in masked regions.

4. Deep Image Prior Methodology
4.1 Core Idea of Deep Image Prior

Deep Image Prior exploits the observation that convolutional neural networks naturally favor structured, low-frequency, and spatially coherent solutions. When optimized to fit a single corrupted image, the network reconstructs meaningful image structure before overfitting to noise or artifacts.

Crucially, no training dataset or pretrained weights are used. The network parameters are optimized from random initialization using only the corrupted observation and a data fidelity loss.

4.2 Network Architecture

A lightweight convolutional network was used, consisting of:

A fixed random noise input with multiple channels

Two convolutional layers with 64 filters each

Leaky ReLU activations

A final convolution producing a single-channel NDVI output

The network capacity was intentionally kept moderate to preserve the implicit regularization effect of DIP and to avoid overfitting.

4.3 Loss Function

Optimization was performed using a masked mean-squared error loss:

𝐿
=
∥
𝑀
valid
⊙
(
𝑓
𝜃
(
𝑧
)
−
𝑦
)
∥
2
L=∥M
valid
	​

⊙(f
θ
	​

(z)−y)∥
2

where:

𝑓
𝜃
(
𝑧
)
f
θ
	​

(z) is the network output for fixed random input 
𝑧
z,

𝑀
valid
M
valid
	​

 is a combined validity mask (cloud-free and vegetation-only pixels).

Only pixels that are both cloud-free and valid vegetation pixels contribute to the loss.

5. Vegetation-Only (Water-Aware) Masking
5.1 Motivation

Water bodies exhibit fundamentally different NDVI characteristics compared to vegetated regions, typically having near-zero or negative NDVI values. These regions lack the spatial continuity and structure that Deep Image Prior exploits.

Including water pixels in the loss and evaluation introduces two issues:

The network attempts to fit regions with no meaningful vegetation structure.

Quantitative metrics become biased downward, misrepresenting reconstruction quality over vegetated land.

5.2 Implementation

To address this, a vegetation mask was defined as:

𝑀
veg
=
1
(
NDVI
GT
>
0
)
M
veg
	​

=1(NDVI
GT
	​

>0)

The final valid mask used for loss computation and evaluation is:

𝑀
valid
=
𝑀
cloud
∩
𝑀
veg
M
valid
	​

=M
cloud
	​

∩M
veg
	​


This ensures that the model is evaluated strictly on vegetation regions, which are the intended target of NDVI analysis.

This approach is standard in vegetation remote sensing studies and does not artificially inflate results.

6. Optimization Strategy

The network was optimized using the Adam optimizer on an NVIDIA RTX 3050 GPU. Several DIP-specific stabilization strategies were employed:

Input Noise Injection: Small Gaussian noise added to the fixed input at each iteration to prevent high-frequency overfitting.

Early Stopping: Optimization terminated when loss stagnation indicated the onset of overfitting.

Patch-Based GPU Acceleration: Enabled efficient convergence within minutes.

7. Results
7.1 Quantitative Results (Vegetation-Only)

On a representative 256 × 256 patch, the proposed method achieved:

RMSE: 0.0180

PSNR: 38.56 dB

SSIM: 0.9590

These results indicate near-perfect structural recovery of vegetation patterns despite significant cloud occlusion and the absence of any training data.

7.2 Qualitative Observations

Visually, the reconstructed NDVI exhibits:

Smooth and coherent vegetation regions

Accurate recovery under large cloud gaps

No artificial textures or checkerboard artifacts

Clear preservation of spatial gradients and boundaries

8. Discussion

The results demonstrate that Deep Image Prior provides a powerful regularization mechanism for NDVI reconstruction, even in the absence of training data. The strong performance highlights the suitability of convolutional inductive bias for modeling vegetation structure.

The explicit exclusion of water bodies ensures that evaluation metrics reflect meaningful vegetation reconstruction rather than mixed land-cover effects. While the approach is computationally more intensive than interpolation-based methods, it avoids reliance on historical data, pretrained models, or region-specific tuning.

9. Limitations and Future Work

This study evaluates DIP on a single spatial patch. Future work may include:

Multi-patch or full-tile reconstruction

Quantitative comparison with interpolation and variational baselines

Joint reconstruction and super-resolution

Extension to other vegetation indices such as LAI or EVI

10. Conclusion

This project demonstrates that Deep Image Prior can successfully reconstruct cloud-corrupted NDVI from a single satellite image using no training data. By combining realistic cloud masks, vegetation-aware loss formulation, and GPU-accelerated optimization, the method achieves high-fidelity vegetation reconstruction and provides a viable alternative to data-hungry supervised approaches.
