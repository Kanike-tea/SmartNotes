"""
pdf_processor.py

End-to-end PDF processing pipeline for SmartNotes.
Handles PDF extraction, OCR, subject classification, and file organization.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import cv2

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("pdf2image not installed. PDF processing disabled. Install with: pip install pdf2image")

from .recognize import OCRLMInference

# Import subject classifier from preprocessing module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from preprocessing.subject_classifier import classify_subject


@dataclass
class ProcessedPage:
    """Represents a single processed page from a PDF."""
    page_number: int
    source_pdf: str
    extracted_text: str
    subject_code: str
    subject_name: str
    confidence: float
    keywords_matched: List[str]
    timestamp: str
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PDFProcessingResult:
    """Represents the complete result of processing a PDF."""
    pdf_name: str
    total_pages: int
    processed_pages: int
    pages: List[ProcessedPage]
    errors: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pdf_name": self.pdf_name,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "pages": [p.to_dict() for p in self.pages],
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


class PDFProcessor:
    """
    Main PDF processor for the SmartNotes system.
    
    Handles:
    - PDF to image extraction
    - OCR on each page
    - Subject classification
    - Organized output with metadata
    """
    
    def __init__(
        self,
        use_lm: bool = True,
        device: str = "auto",
        save_images: bool = True,
        save_metadata: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PDF processor.
        
        Args:
            use_lm: Whether to use language model for OCR
            device: Device to use ('cpu', 'cuda', 'mps', 'auto')
            save_images: Whether to save extracted page images
            save_metadata: Whether to save processing metadata
            logger: Logger instance (creates default if None)
        """
        if not PDF_SUPPORT:
            raise ImportError(
                "pdf2image is required for PDF processing. "
                "Install with: pip install pdf2image"
            )
        
        self.logger = logger or self._setup_logger()
        self.ocr_inference = OCRLMInference(use_lm=use_lm, device=device)  # type: ignore
        self.save_images = save_images
        self.save_metadata = save_metadata
        
        self.logger.info(f"PDFProcessor initialized with device: {self.ocr_inference.device}")
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup default logger."""
        logger = logging.getLogger("PDFProcessor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        dpi: int = 200
    ) -> PDFProcessingResult:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save outputs
            dpi: DPI for PDF to image conversion
            
        Returns:
            PDFProcessingResult with processed pages and metadata
        """
        pdf_path_obj = Path(pdf_path)  # type: ignore
        output_dir_obj = Path(output_dir)  # type: ignore
        
        if not pdf_path_obj.exists():  # type: ignore
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        output_dir_obj.mkdir(parents=True, exist_ok=True)  # type: ignore
        
        self.logger.info(f"Processing PDF: {pdf_path_obj.name}")  # type: ignore
        
        # Extract images from PDF
        try:
            images = convert_from_path(str(pdf_path), dpi=dpi)  # type: ignore
            self.logger.info(f"Extracted {len(images)} pages from PDF")
        except Exception as e:
            self.logger.error(f"Failed to extract PDF: {e}")
            raise
        
        # Process each page
        processed_pages: List[ProcessedPage] = []
        errors: List[str] = []
        
        for page_num, image in enumerate(images, 1):
            try:
                page_result = self._process_page(
                    image, pdf_path_obj.name, page_num, output_dir_obj  # type: ignore
                )
                processed_pages.append(page_result)
                self.logger.info(
                    f"Page {page_num}/{len(images)}: "
                    f"{page_result.subject_code} "
                    f"(confidence: {page_result.confidence:.2f})"
                )
            except Exception as e:
                error_msg = f"Page {page_num}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        result = PDFProcessingResult(
            pdf_name=pdf_path_obj.name,  # type: ignore
            total_pages=len(images),
            processed_pages=len(processed_pages),
            pages=processed_pages,
            errors=errors,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(
            f"PDF processing complete: "
            f"{len(processed_pages)}/{len(images)} pages processed"
        )
        
        return result
    
    def _process_page(
        self,
        image,  # PIL Image
        pdf_name: str,
        page_num: int,
        output_dir
    ) -> ProcessedPage:
        """
        Process a single page image.
        
        Args:
            image: PIL Image object from PDF page
            pdf_name: Name of source PDF
            page_num: Page number in PDF
            output_dir: Output directory for files
            
        Returns:
            ProcessedPage with OCR and classification results
        """
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Save page image if requested
        image_path = None
        if self.save_images:
            image_path = output_dir / f"page_{page_num:04d}.png"  # type: ignore
            cv2.imwrite(str(image_path), image_cv)
        
        # Run OCR
        try:
            # OCR inference expects image path or array
            # Create temporary file for OCR
            temp_image_path = output_dir / f"temp_page_{page_num}.png"  # type: ignore
            cv2.imwrite(str(temp_image_path), image_cv)
            
            # Run OCR (inference returns predictions and metrics)
            ocr_results = self.ocr_inference.infer_single(str(temp_image_path))  # type: ignore
            extracted_text = ocr_results.get("prediction", "")
            
            # Cleanup temp file
            temp_image_path.unlink()  # type: ignore
        except Exception as e:
            self.logger.warning(f"OCR failed for page {page_num}: {e}")
            extracted_text = ""
        
        # Classify subject
        subject_code, keywords, confidence = classify_subject(extracted_text)
        
        # Extract subject code from full subject name if available
        if " - " in subject_code:
            subject_code_short = subject_code.split(" - ")[0]
        else:
            subject_code_short = subject_code
        
        page_result = ProcessedPage(
            page_number=page_num,
            source_pdf=pdf_name,
            extracted_text=extracted_text,
            subject_code=subject_code_short,
            subject_name=subject_code,
            confidence=confidence,
            keywords_matched=keywords,
            timestamp=datetime.now().isoformat(),
            image_path=str(image_path) if image_path else None
        )
        
        return page_result
    
    def process_batch(
        self,
        pdf_dir: str,
        output_dir: str,
        pattern: str = "*.pdf",
        organize_by_subject: bool = True
    ) -> Dict[str, PDFProcessingResult]:
        """
        Process multiple PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory for outputs
            pattern: File pattern to match (default: *.pdf)
            organize_by_subject: Whether to organize outputs by subject folder
            
        Returns:
            Dictionary mapping PDF names to processing results
        """
        pdf_dir_obj = Path(pdf_dir)  # type: ignore
        output_dir_obj = Path(output_dir)  # type: ignore
        
        if not pdf_dir_obj.exists():  # type: ignore
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
        # Find PDFs
        pdfs = list(pdf_dir_obj.glob(pattern))  # type: ignore
        self.logger.info(f"Found {len(pdfs)} PDFs in {pdf_dir}")
        
        if not pdfs:
            self.logger.warning(f"No PDFs matching pattern '{pattern}' in {pdf_dir}")
            return {}
        
        results = {}
        for pdf_path in pdfs:
            try:
                result = self.process_pdf(str(pdf_path), str(output_dir))
                results[pdf_path.name] = result
                
                # Organize by subject if requested
                if organize_by_subject:
                    self._organize_by_subject(result, output_dir_obj)  # type: ignore
                
                # Save metadata
                if self.save_metadata:
                    self._save_metadata(result, output_dir_obj)  # type: ignore
                    
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_path.name}: {e}")
        
        return results
    
    def _organize_by_subject(
        self,
        result: PDFProcessingResult,
        output_dir: Path
    ) -> None:
        """
        Organize processed pages into subject-based folders.
        
        Args:
            result: Processing result with page data
            output_dir: Base output directory
        """
        # Group pages by subject
        subjects = {}
        for page in result.pages:
            subject = page.subject_code
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(page)
        
        # Create subject folders and copy/move files
        for subject, pages in subjects.items():
            subject_dir = output_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subject metadata file
            subject_metadata = {
                "subject_code": subject,
                "pages": [p.to_dict() for p in pages],
                "total_pages": len(pages),
                "timestamp": datetime.now().isoformat(),
            }
            
            metadata_path = subject_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(subject_metadata, f, indent=2)
            
            self.logger.info(f"Organized {len(pages)} pages into {subject}/")
    
    def _save_metadata(
        self,
        result: PDFProcessingResult,
        output_dir: Path
    ) -> None:
        """
        Save processing metadata to JSON file.
        
        Args:
            result: Processing result
            output_dir: Directory for metadata file
        """
        metadata_path = output_dir / "processing_metadata.json"
        
        # Load existing metadata if available
        existing = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    existing = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load existing metadata: {e}")
        
        # Add new result
        if "results" not in existing:
            existing["results"] = {}
        
        existing["results"][result.pdf_name] = result.to_dict()
        existing["updated"] = datetime.now().isoformat()
        
        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(existing, f, indent=2)
    
    def generate_summary_report(
        self,
        results: Dict[str, PDFProcessingResult],
        output_file: str
    ) -> None:
        """
        Generate a summary report of batch processing.
        
        Args:
            results: Dictionary of processing results
            output_file: Path to save summary report
        """
        summary = {
            "total_pdfs": len(results),
            "total_pages": sum(r.total_pages for r in results.values()),
            "successfully_processed": sum(r.processed_pages for r in results.values()),
            "failed_pages": sum(len(r.errors) for r in results.values()),
            "timestamp": datetime.now().isoformat(),
            "by_subject": {},
            "by_confidence": {
                "high": 0,  # confidence >= 0.5
                "medium": 0,  # 0.25 <= confidence < 0.5
                "low": 0,  # confidence < 0.25
            },
            "details": [],
        }
        
        # Analyze results
        for pdf_name, result in results.items():
            pdf_summary = {
                "pdf_name": pdf_name,
                "total_pages": result.total_pages,
                "processed": result.processed_pages,
                "errors": len(result.errors),
                "subjects": {},
            }
            
            # Group by subject
            for page in result.pages:
                subj = page.subject_code
                if subj not in pdf_summary["subjects"]:
                    pdf_summary["subjects"][subj] = 0
                pdf_summary["subjects"][subj] += 1
                
                # Track confidence distribution
                if page.confidence >= 0.5:
                    summary["by_confidence"]["high"] += 1
                elif page.confidence >= 0.25:
                    summary["by_confidence"]["medium"] += 1
                else:
                    summary["by_confidence"]["low"] += 1
                
                # Track subjects globally
                if subj not in summary["by_subject"]:
                    summary["by_subject"][subj] = 0
                summary["by_subject"][subj] += 1
            
            summary["details"].append(pdf_summary)
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary report saved to {output_file}")
        self._print_summary(summary)
    
    @staticmethod
    def _print_summary(summary: Dict) -> None:
        """Print formatted summary report."""
        print("\n" + "="*70)
        print("PDF PROCESSING SUMMARY REPORT")
        print("="*70)
        print(f"Total PDFs: {summary['total_pages']}")
        print(f"Total Pages: {summary['total_pages']}")
        print(f"Successfully Processed: {summary['successfully_processed']}")
        print(f"Failed Pages: {summary['failed_pages']}")
        print(f"\nConfidence Distribution:")
        print(f"  High (≥0.5):     {summary['by_confidence']['high']}")
        print(f"  Medium (0.25-0.5): {summary['by_confidence']['medium']}")
        print(f"  Low (<0.25):     {summary['by_confidence']['low']}")
        print(f"\nPages by Subject:")
        for subject, count in sorted(summary["by_subject"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {subject}: {count}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor(use_lm=True)
    
    # Process single PDF
    # result = processor.process_pdf(
    #     "path/to/pdf.pdf",
    #     "output_dir"
    # )
    # print(result)
    
    # Process batch
    # results = processor.process_batch(
    #     "path/to/pdf_dir",
    #     "output_dir",
    #     organize_by_subject=True
    # )
    # processor.generate_summary_report(results, "summary.json")
