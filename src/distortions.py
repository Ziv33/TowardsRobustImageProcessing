"""
src/distortions.py

Implements systematic image degradations: Gaussian Noise, JPEG Compression,
and Low-Light Simulation. All mathematical transformations are executed using 
high-precision float64 representations to prevent clipping/overflow issues, 
then cast safely back to standard uint8 bounds.
"""

import numpy as np
import cv2

def apply_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies additive Gaussian noise to the input image.
    
    Args:
        image: np.ndarray, input image in RGB format (H, W, 3), range [0, 255]
        sigma: Standard deviation of the Gaussian noise distribution. Higher
               values indicate greater noise levels.
               
    Returns:
        np.ndarray: Noisy image in RGB format (uint8)
    """
    if sigma <= 0:
        return image.copy()
        
    # Convert image to high-precision float64 to prevent overflow/underflow
    img_f64 = image.astype(np.float64)
    
    # Generate Gaussian noise matching the exact dimensions of the image
    # Mean is set to 0, standard deviation is set to sigma
    noise = np.random.normal(loc=0.0, scale=sigma, size=img_f64.shape)
    
    # Apply noise and clamp values to standard [0, 255] boundaries
    noisy_img = img_f64 + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """
    Simulates JPEG compression artifacts by encoding and decoding the image.
    
    Args:
        image: np.ndarray, input image in RGB format (H, W, 3), range [0, 255]
        quality: Compression quality factor [1, 100]. Lower values mean 
                 severe artifacting and blockiness.
                 
    Returns:
        np.ndarray: Compressed image in RGB format (uint8)
    """
    quality = int(np.clip(quality, 1, 100))
    
    # OpenCV's JPEG encoder expects BGR color format
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Set the compression quality parameter
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # Encode the image to memory buffer
    success, encoded_img = cv2.imencode('.jpg', bgr, encode_param)
    if not success:
        return image.copy()
        
    # Decode the image back from memory
    decoded_bgr = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    # Convert back to RGB format
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def apply_low_light(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Simulates low-light conditions using non-linear gamma correction.
    
    Args:
        image: np.ndarray, input image in RGB format (H, W, 3), range [0, 255]
        gamma: Exponent value. Values > 1.0 reduce mid-tone intensities,
               simulating undereposure.
               
    Returns:
        np.ndarray: Darkened image in RGB format (uint8)
    """
    if gamma <= 0:
        gamma = 1.0
        
    # Convert pixel values to [0.0, 1.0] range using float64 representation
    img_f64 = image.astype(np.float64) / 255.0
    
    # Apply power-law gamma transformation
    # Values above 1.0 curve the pixel values downwards non-linearly
    darkened = np.power(img_f64, gamma) * 255.0
    
    # Clamp values to valid pixel boundaries and convert back to uint8
    return np.clip(darkened, 0, 255).astype(np.uint8)