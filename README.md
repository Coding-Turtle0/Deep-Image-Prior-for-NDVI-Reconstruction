# Deep-Image-Prior-for-NDVI-Reconstruction
This repository presents a Deep Image Prior (DIP)–based approach for reconstructing cloud-corrupted NDVI using real MODIS satellite data. The method relies entirely on the implicit regularization of an untrained convolutional neural network and does not require any training data, pretrained weights, or historical imagery.

The implementation demonstrates that meaningful vegetation structure can be recovered from a single corrupted observation by directly optimizing network parameters on the target image.

Project Motivation

Cloud contamination is a persistent problem in optical remote sensing and significantly limits the usability of vegetation indices such as NDVI. Conventional gap-filling techniques typically rely on temporal compositing, spatial interpolation, or supervised learning models trained on large historical datasets.

This project explores an alternative paradigm: using Deep Image Prior to recover missing NDVI values directly from a single cloud-corrupted image, without any external data. The goal is to assess whether architectural bias alone is sufficient to reconstruct vegetation structure under realistic cloud conditions.

Dataset Description
Source Product

The dataset is derived from the MODIS MOD13Q1 product, which provides 16-day composite NDVI at 250 m spatial resolution. MOD13Q1 is widely used for vegetation monitoring due to its global coverage and radiometric consistency.

Extracted Layers

The following layers were extracted from the MOD13Q1 HDF file:

NDVI (scaled using the official MODIS scale factor)

Pixel Reliability layer

The pixel reliability layer was used to construct a realistic cloud mask based on MODIS quality flags.

Cloud Masking Strategy

Pixels marked as cloud, snow/ice, or marginal quality in the MODIS reliability layer were treated as missing data. Only pixels labeled as “good quality” were retained as valid observations.

This approach ensures that the forward model reflects real satellite cloud contamination rather than synthetic masking patterns.

Vegetation-Only Masking (Water Handling)

Water bodies exhibit fundamentally different NDVI behavior compared to vegetated land. NDVI values over water are typically near zero or negative and do not follow the spatial continuity assumptions that Deep Image Prior exploits.

Including water pixels in the loss function and evaluation leads to degraded reconstruction quality and misleading performance metrics. To address this, vegetation-only masking was applied:

Pixels with NDVI ≤ 0 were excluded from both optimization and evaluation.

Reconstruction quality was assessed only over vegetated regions.

This masking strategy is standard practice in vegetation remote sensing studies and ensures that reported metrics reflect meaningful vegetation reconstruction.

Method Overview

A fixed random noise tensor is provided as input to an untrained convolutional neural network.

Network parameters are optimized directly on the cloud-corrupted NDVI image.

The loss function is computed only over cloud-free, vegetation-only pixels.

Input noise injection and early stopping are used to prevent overfitting.

Optimization is accelerated using GPU computation.

The method relies solely on architectural bias and does not use any training data.

Implementation Details

Framework: MATLAB Deep Learning Toolbox

Hardware: NVIDIA RTX 3050 GPU

Processing strategy: Patch-based reconstruction

Patch size: 256 × 256

Network depth: Moderate (64 feature channels)

Optimization: Adam optimizer with early stopping

Stabilization: Input noise injection

Patch-based processing is used to reduce memory requirements and improve optimization stability. This approach is standard in remote sensing image restoration.

Results

Reconstruction was performed on a single 256 × 256 NDVI patch with real cloud coverage.

Quantitative Results (Vegetation-Only)

RMSE: 0.0180

PSNR: 38.56 dB

SSIM: 0.9590

These results indicate near-perfect structural recovery of vegetation patterns despite significant cloud occlusion and the absence of any training data.

Qualitative Observations

Smooth and spatially coherent vegetation regions

Accurate recovery beneath cloud gaps

No artificial textures or checkerboard artifacts

Preservation of large-scale vegetation gradients

Why Deep Image Prior Works Well for NDVI

NDVI fields are spatially smooth and structured

Convolutional architectures naturally favor such structure

The network learns large-scale patterns before fitting fine-scale noise

Early stopping prevents overfitting to corrupted observations

These properties align well with the characteristics of vegetation indices.

Limitations

Reconstruction is demonstrated on a single spatial patch

Computational cost is higher than simple interpolation methods

Water regions are not reconstructed due to fundamentally different NDVI behavior

Future Work

Multi-patch and full-tile reconstruction

Comparison with interpolation and variational baselines

Joint cloud removal and super-resolution

Extension to LAI and other vegetation indices

Integration with multi-spectral reflectance data
10. Conclusion

This project demonstrates that Deep Image Prior can successfully reconstruct cloud-corrupted NDVI from a single satellite image using no training data. By combining realistic cloud masks, vegetation-aware loss formulation, and GPU-accelerated optimization, the method achieves high-fidelity vegetation reconstruction and provides a viable alternative to data-hungry supervised approaches.

References

Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018).
Deep Image Prior.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
