"""
PDF Processor for extracting pages as images from handwritten and printed notes.

This module provides utilities to convert PDF documents into images that can be
used for OCR training and inference. Supports both handwritten and printed notes.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path  # type: ignore
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None  # type: ignore
    logger.warning(
        "pdf2image not installed. Install it with: pip install pdf2image\n"
        "You also need: brew install poppler (on macOS) or apt-get install poppler-utils (on Linux)"
    )

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore
    logger.warning("opencv-python not installed. Install it with: pip install opencv-python")


class PDFProcessor:
    """
    Convert PDF documents to images for OCR training.
    
    Attributes:
        dpi: Resolution for PDF conversion (default 150)
        max_size: Maximum image dimension for resizing (None = no resize)
    """
    
    def __init__(self, dpi: int = 150, max_size: Optional[int] = None) -> None:
        """
        Initialize PDF processor.
        
        Args:
            dpi: Resolution for PDF conversion (higher = better quality, slower)
            max_size: Maximum image dimension. If set, images larger than this
                     will be resized to fit within max_size x max_size (maintains aspect ratio)
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError(
                "pdf2image not installed. Install with:\n"
                "  pip install pdf2image\n"
                "Then install poppler:\n"
                "  macOS: brew install poppler\n"
                "  Linux: apt-get install poppler-utils"
            )
        
        self.dpi = dpi
        self.max_size = max_size
        if convert_from_path is None:
            raise RuntimeError(
                "pdf2image not installed. Install with:\n"
                "  pip install pdf2image\n"
                "Then install poppler:\n"
                "  macOS: brew install poppler\n"
                "  Linux: apt-get install poppler-utils"
            )
    
    def pdf_to_images(
        self, 
        pdf_path: str, 
        start_page: int = 1, 
        end_page: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Convert PDF pages to numpy arrays (grayscale images).
        
        Args:
            pdf_path: Path to PDF file
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (inclusive). If None, convert all pages.
            
        Returns:
            List of grayscale numpy arrays (uint8, shape HxW)
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If conversion fails
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            logger.debug(f"Converting PDF: {pdf_path} (dpi={self.dpi})")
            
            # Convert PDF to images (PIL format)
            # Build kwargs to avoid passing None for last_page (convert_from_path expects an int)
            kwargs = {
                "dpi": self.dpi,
                "first_page": start_page,
                "fmt": "ppm"  # Use PPM format for better quality
            }
            if end_page is not None:
                kwargs["last_page"] = end_page

            images = convert_from_path(  # type: ignore
                pdf_path,
                **kwargs
            )
            
            # Convert PIL images to grayscale numpy arrays
            gray_images = []
            for img in images:
                # Convert PIL Image to numpy array
                img_array = np.array(img)
                
                # Convert RGB to grayscale
                if len(img_array.shape) == 3:
                    if CV2_AVAILABLE and cv2 is not None:
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # type: ignore
                    else:
                        gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                else:
                    gray = img_array
                
                # Resize if needed
                if self.max_size is not None:
                    gray = self._resize_image(gray)
                
                gray_images.append(gray)
            
            logger.debug(f"Successfully converted {len(gray_images)} pages")
            return gray_images
        
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF {pdf_path}: {e}")
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to fit within max_size while maintaining aspect ratio.
        
        Args:
            img: Input image array
        
        Returns:
            Resized image array
        """
        if self.max_size is None:
            return img
        
        h, w = img.shape[:2]
        
        if h > self.max_size or w > self.max_size:
            scale = self.max_size / max(h, w)  # type: ignore
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            if CV2_AVAILABLE and cv2 is not None:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)  # type: ignore
            else:
                # Fallback: simple numpy-based resize (not ideal)
                logger.warning("Using fallback resize (cv2 not available). Consider installing opencv-python")
                img = np.array(img)  # Will be approximate
        
        return img
    
    def extract_text_regions(
        self, 
        img: np.ndarray, 
        line_height_range: Tuple[int, int] = (20, 100)
    ) -> List[np.ndarray]:
        """
        Extract potential text regions from a document image.
        
        Simple approach: Find horizontal line regions based on contours/thresholding.
        
        Args:
            img: Grayscale image
            line_height_range: Expected line height range (min, max)
            
        Returns:
            List of cropped image regions containing text
        """
        if not CV2_AVAILABLE or cv2 is None:
            logger.warning("opencv-python required for extract_text_regions")
            return [img]
        
        try:
            # Threshold image
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # type: ignore
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore
            
            # Extract bounding boxes for potential text lines
            regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)  # type: ignore
                
                # Filter by height
                if line_height_range[0] <= h <= line_height_range[1]:
                    # Crop with padding
                    padding = 5
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(img.shape[1], x + w + padding)
                    y2 = min(img.shape[0], y + h + padding)
                    
                    region = img[y1:y2, x1:x2]
                    if region.shape[0] > 0 and region.shape[1] > 0:
                        regions.append(region)
            
            return regions if regions else [img]
        
        except Exception as e:
            logger.warning(f"Error extracting text regions: {e}")
            return [img]


def process_notes_directory(
    notes_dir: str,
    output_dir: str,
    dpi: int = 150,
    extract_regions: bool = False
) -> int:
    """
    Process all PDF files in a directory and save extracted images.
    
    Args:
        notes_dir: Directory containing PDF files
        output_dir: Directory to save extracted images
        dpi: Resolution for PDF conversion
        extract_regions: Whether to extract individual text regions
        
    Returns:
        Total number of images extracted
    """
    if not os.path.exists(notes_dir):
        logger.error(f"Notes directory not found: {notes_dir}")
        return 0
    
    if not CV2_AVAILABLE or cv2 is None:
        logger.error("opencv-python required for saving images")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    processor = PDFProcessor(dpi=dpi)
    total_images = 0
    
    # Find all PDF files recursively
    pdf_files = list(Path(notes_dir).rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {notes_dir}")
    
    for pdf_path in pdf_files:
        try:
            logger.debug(f"Processing: {pdf_path}")
            
            images = processor.pdf_to_images(str(pdf_path))
            
            # Create subdirectory based on PDF filename
            pdf_name = pdf_path.stem
            pdf_output_dir = os.path.join(output_dir, pdf_name)
            os.makedirs(pdf_output_dir, exist_ok=True)
            
            # Save images
            for page_idx, img in enumerate(images):
                if extract_regions:
                    regions = processor.extract_text_regions(img)
                    for region_idx, region in enumerate(regions):
                        filename = f"{pdf_name}_page{page_idx:03d}_region{region_idx:02d}.png"
                        filepath = os.path.join(pdf_output_dir, filename)
                        cv2.imwrite(filepath, region)  # type: ignore
                        total_images += 1
                else:
                    filename = f"{pdf_name}_page{page_idx:03d}.png"
                    filepath = os.path.join(pdf_output_dir, filename)
                    cv2.imwrite(filepath, img)  # type: ignore
                    total_images += 1
            
            logger.info(f"Extracted {len(images)} pages from {pdf_path.name}")
        
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            continue
    
    logger.info(f"Total images extracted: {total_images}")
    return total_images


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract images from PDF notes")
    parser.add_argument("--input", required=True, help="Input directory with PDFs")
    parser.add_argument("--output", required=True, help="Output directory for images")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for conversion")
    parser.add_argument("--extract-regions", action="store_true", help="Extract text regions")
    
    args = parser.parse_args()
    
    total = process_notes_directory(
        args.input,
        args.output,
        dpi=args.dpi,
        extract_regions=args.extract_regions
    )
    print(f"Successfully extracted {total} images to {args.output}")
