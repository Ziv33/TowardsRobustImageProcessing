"""
src/enhancements.py

Classical image-restoration and pre-processing techniques.
Upgraded to use adaptive parameters based on estimated or known degradation 
levels to prevent over-smoothing of clean or lightly distorted inputs.
"""

import numpy as np
import cv2

def restore_noise_bilateral(image: np.ndarray, estimated_sigma: float = 30.0) -> np.ndarray:
    """
    Applies adaptive bilateral filtering to suppress high-frequency noise.
    Filters scale window size and color standard deviation to avoid over-blurring
    fine features in low-noise conditions.
    
    Args:
        image: np.ndarray, input image in RGB format (H, W, 3), range [0, 255]
        estimated_sigma: Estimated or known standard deviation of the Gaussian noise.
        
    Returns:
        np.ndarray: Denoised image in RGB format (uint8)
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Adaptive parameter selection
    if estimated_sigma <= 5.0:
        d = 5
        sigma_color = 15
        sigma_space = 15
    elif estimated_sigma <= 20.0:
        d = 7
        sigma_color = 50
        sigma_space = 50
    else:
        d = 11
        sigma_color = 120
        sigma_space = 120
        
    denoised_bgr = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)


def restore_jpeg_deblocking(image: np.ndarray, estimated_quality: int = 20) -> np.ndarray:
    """
    Applies adaptive edge-preserving filtering to smooth blocky boundaries
    induced by JPEG lossy compression.
    
    Args:
        image: np.ndarray, input image in RGB format (H, W, 3), range [0, 255]
        estimated_quality: Estimated or known JPEG quality factor [1, 100].
        
    Returns:
        np.ndarray: Deblocked image in RGB format (uint8)
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Scale filter size based on quality factor
    if estimated_quality >= 80:
        d = 3
        sigma_color = 15
        sigma_space = 15
    elif estimated_quality >= 40:
        d = 5
        sigma_color = 45
        sigma_space = 45
    else:
        d = 9
        sigma_color = 90
        sigma_space = 90
        
    restored_bgr = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)


def restore_lowlight_clahe(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """
    Applies a Dual-Stage Low-Light Restoration:
    1. Global Gamma Expansion (gamma = 0.40) to elevate shadow regions.
    2. Local Contrast Optimization (CLAHE in LAB space) to recover structural edges.
    
    Args:
        image: np.ndarray, input image in RGB format (H, W, 3), range [0, 255]
        clip_limit: Threshold for contrast limiting in CLAHE.
        
    Returns:
        np.ndarray: Enhanced image in RGB format (uint8)
    """
    img_f64 = image.astype(np.float64) / 255.0
    boosted_f64 = np.power(img_f64, 0.40) * 255.0
    boosted_img = np.clip(boosted_f64, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(boosted_img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)