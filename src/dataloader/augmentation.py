"""
Data augmentation for OCR training.
Improves model robustness and generalization.

Augmentation techniques:
- Rotation: Small rotations (±3-5 degrees) to simulate paper angle
- Brightness: Brightness variations to handle different lighting
- Contrast: Contrast adjustments for different paper/ink combinations
- Noise: Gaussian noise to simulate scanner/camera noise
- Blur: Slight blur to simulate focus issues
- Elastic: Paper warping to simulate real-world variations
"""

import cv2
import numpy as np
import random
from typing import Tuple, Optional


class OCRAugmentation:
    """
    Augmentation pipeline for OCR images.
    Applies transformations that preserve text readability.
    
    All augmentations are designed to:
    - Maintain text legibility
    - Simulate real-world image variations
    - Not distort characters beyond recognition
    """
    
    def __init__(
        self,
        rotation_range: float = 5.0,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.7, 1.3),
        noise_probability: float = 0.3,
        blur_probability: float = 0.2,
        elastic_probability: float = 0.1,
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            rotation_range: Max rotation in degrees (±)
            brightness_range: Brightness factor range (min, max)
            contrast_range: Contrast factor range (min, max)
            noise_probability: Probability of adding noise
            blur_probability: Probability of adding blur
            elastic_probability: Probability of elastic deformation
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_probability = noise_probability
        self.blur_probability = blur_probability
        self.elastic_probability = elastic_probability
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.
        
        Args:
            image: Input image (grayscale, H×W)
            
        Returns:
            Augmented image
        """
        if image is None or image.size == 0:
            return image
        
        # 1. Small rotation (preserves readability)
        if random.random() < 0.5:
            image = self._rotate(image)
        
        # 2. Brightness adjustment
        if random.random() < 0.6:
            image = self._adjust_brightness(image)
        
        # 3. Contrast adjustment
        if random.random() < 0.6:
            image = self._adjust_contrast(image)
        
        # 4. Gaussian noise
        if random.random() < self.noise_probability:
            image = self._add_noise(image)
        
        # 5. Slight blur (simulates poor camera/scanner)
        if random.random() < self.blur_probability:
            image = self._add_blur(image)
        
        # 6. Elastic deformation (simulates paper warping)
        if random.random() < self.elastic_probability:
            image = self._elastic_transform(image)
        
        return image
    
    def _rotate(self, image: np.ndarray) -> np.ndarray:
        """
        Small rotation (±5 degrees).
        
        Args:
            image: Input image
            
        Returns:
            Rotated image
        """
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust brightness.
        
        Args:
            image: Input image
            
        Returns:
            Brightness-adjusted image
        """
        factor = random.uniform(*self.brightness_range)
        adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
        return adjusted
    
    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust contrast.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-adjusted image
        """
        factor = random.uniform(*self.contrast_range)
        mean = np.mean(image)
        adjusted = np.clip(
            (image - mean) * factor + mean, 0, 255
        ).astype(np.uint8)
        return adjusted
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise.
        
        Args:
            image: Input image
            
        Returns:
            Noisy image
        """
        noise = np.random.normal(0, 10, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def _add_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Add slight Gaussian blur.
        
        Args:
            image: Input image
            
        Returns:
            Blurred image
        """
        kernel_size = random.choice([3, 5])
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def _elastic_transform(
        self, 
        image: np.ndarray,
        alpha: float = 10,
        sigma: float = 3
    ) -> np.ndarray:
        """
        Elastic deformation (simulates paper warping).
        
        Args:
            image: Input image
            alpha: Deformation strength
            sigma: Smoothness of deformation
            
        Returns:
            Deformed image
        """
        h, w = image.shape
        
        # Random displacement field
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1),
            (0, 0),
            sigma
        ) * alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1),
            (0, 0),
            sigma
        ) * alpha
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply transformation
        deformed = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return deformed


def get_augmentation(training: bool = True) -> Optional[OCRAugmentation]:
    """
    Get augmentation pipeline for training or validation.
    
    Args:
        training: If True, returns augmentation; if False, returns None
        
    Returns:
        OCRAugmentation instance or None
    """
    if training:
        return OCRAugmentation(
            rotation_range=3.0,  # Conservative for OCR
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            noise_probability=0.2,
            blur_probability=0.1,
            elastic_probability=0.05,
        )
    else:
        return None


if __name__ == "__main__":
    # Test augmentation
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if image_path.exists():
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            aug = OCRAugmentation()
            
            print("Applying augmentations...")
            for i in range(5):
                augmented = aug(img)
                output_path = image_path.parent / f"augmented_{i}.png"
                cv2.imwrite(str(output_path), augmented)
                print(f"  Saved: {output_path}")
        else:
            print(f"Image not found: {image_path}")
    else:
        print("Usage: python augmentation.py <image_path>")
        print("Creates 5 augmented versions of the image")
