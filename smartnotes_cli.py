#!/usr/bin/env python3
"""
SmartNotes End-to-End CLI Application

Complete workflow for processing academic notes:
  PDF → Extract → OCR → Classify Subject → Organize by Course

Usage:
    python smartnotes_cli.py --pdf document.pdf --output ./results
    python smartnotes_cli.py --batch ./pdfs --output ./results --organize
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.pdf_processor import PDFProcessor, PDFProcessingResult
from config import Config


class SmartNotesApp:
    """Main SmartNotes application."""
    
    def __init__(
        self,
        use_lm: bool = True,
        device: str = "auto",
        verbose: bool = False
    ):
        """
        Initialize SmartNotes app.
        
        Args:
            use_lm: Use language model for OCR
            device: Device to use (cpu/cuda/mps/auto)
            verbose: Enable verbose logging
        """
        self.logger = self._setup_logger(verbose)
        self.processor = PDFProcessor(
            use_lm=use_lm,
            device=device,
            save_images=True,
            save_metadata=True,
            logger=self.logger
        )
    
    @staticmethod
    def _setup_logger(verbose: bool = False) -> logging.Logger:
        """Setup application logger."""
        logger = logging.getLogger("SmartNotes")
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            
            # Formatter
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            # Set level
            level = logging.DEBUG if verbose else logging.INFO
            console_handler.setLevel(level)
            logger.setLevel(level)
            
            logger.addHandler(console_handler)
        
        return logger
    
    def process_single_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        organize: bool = True
    ) -> Optional[PDFProcessingResult]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF
            output_dir: Output directory
            organize: Whether to organize by subject
            
        Returns:
            PDFProcessingResult or None if failed
        """
        try:
            self.logger.info(f"Processing PDF: {pdf_path}")
            
            result = self.processor.process_pdf(pdf_path, output_dir)
            
            # Organize by subject if requested
            if organize:
                self.logger.info("Organizing pages by subject...")
                self.processor._organize_by_subject(result, Path(output_dir))
            
            # Save metadata
            self.processor._save_metadata(result, Path(output_dir))
            
            # Print summary
            self._print_result_summary(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process PDF: {e}")
            return None
    
    def process_batch(
        self,
        pdf_dir: str,
        output_dir: str,
        pattern: str = "*.pdf",
        organize: bool = True,
        report: bool = True
    ) -> dict:
        """
        Process multiple PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDFs
            output_dir: Output directory
            pattern: File pattern to match
            organize: Whether to organize by subject
            report: Generate summary report
            
        Returns:
            Dictionary of results
        """
        try:
            self.logger.info(f"Starting batch processing from: {pdf_dir}")
            
            results = self.processor.process_batch(
                pdf_dir,
                output_dir,
                pattern=pattern,
                organize_by_subject=organize
            )
            
            # Generate report if requested
            if report and results:
                report_path = Path(output_dir) / "summary_report.json"
                self.processor.generate_summary_report(results, str(report_path))
                self.logger.info(f"Report saved to: {report_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return {}
    
    @staticmethod
    def _print_result_summary(result: PDFProcessingResult) -> None:
        """Print formatted result summary."""
        print("\n" + "="*70)
        print("PROCESSING RESULT")
        print("="*70)
        print(f"PDF: {result.pdf_name}")
        print(f"Total Pages: {result.total_pages}")
        print(f"Processed: {result.processed_pages}")
        print(f"Failed: {len(result.errors)}")
        
        if result.pages:
            print("\nPages by Subject:")
            subjects = {}
            for page in result.pages:
                subj = page.subject_code
                if subj not in subjects:
                    subjects[subj] = {"count": 0, "avg_conf": 0}
                subjects[subj]["count"] += 1
                subjects[subj]["avg_conf"] += page.confidence
            
            for subj, data in sorted(subjects.items()):
                avg_conf = data["avg_conf"] / data["count"]
                print(f"  {subj}: {data['count']} pages (avg confidence: {avg_conf:.2f})")
        
        if result.errors:
            print(f"\nErrors:")
            for error in result.errors[:5]:  # Show first 5
                print(f"  - {error}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more")
        
        print("="*70 + "\n")
    
    def create_web_summary(
        self,
        output_dir: str,
        title: str = "SmartNotes Processing Summary"
    ) -> None:
        """
        Create a simple HTML summary of processing.
        
        Args:
            output_dir: Directory containing results
            title: HTML page title
        """
        output_dir = Path(output_dir)
        metadata_file = output_dir / "processing_metadata.json"
        
        if not metadata_file.exists():
            self.logger.warning("No processing metadata found")
            return
        
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"Could not load metadata: {e}")
            return
        
        # Generate HTML
        html_content = self._generate_html_summary(metadata, title)
        
        html_path = output_dir / "summary.html"
        with open(html_path, "w") as f:
            f.write(html_content)
        
        self.logger.info(f"HTML summary saved to: {html_path}")
    
    @staticmethod
    def _generate_html_summary(metadata: dict, title: str) -> str:
        """Generate HTML summary."""
        results = metadata.get("results", {})
        
        total_pdfs = len(results)
        total_pages = sum(r["total_pages"] for r in results.values())
        processed = sum(r["processed_pages"] for r in results.values())
        
        # Build subject rows
        subject_rows = ""
        all_subjects = {}
        for pdf_data in results.values():
            for page in pdf_data.get("pages", []):
                subj = page.get("subject_code", "Unknown")
                if subj not in all_subjects:
                    all_subjects[subj] = {"count": 0, "avg_conf": 0}
                all_subjects[subj]["count"] += 1
                all_subjects[subj]["avg_conf"] += page.get("confidence", 0)
        
        for subj in sorted(all_subjects.keys()):
            data = all_subjects[subj]
            avg_conf = data["avg_conf"] / data["count"]
            subject_rows += f"""
            <tr>
                <td>{subj}</td>
                <td>{data['count']}</td>
                <td>{avg_conf:.2f}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-top: 20px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{total_pdfs}</div>
                <div class="stat-label">PDFs Processed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{total_pages}</div>
                <div class="stat-label">Total Pages</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{processed}</div>
                <div class="stat-label">Successfully Processed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{total_pages - processed}</div>
                <div class="stat-label">Failed Pages</div>
            </div>
        </div>
        
        <h2>Pages by Subject</h2>
        <table>
            <tr>
                <th>Subject Code</th>
                <th>Page Count</th>
                <th>Avg Confidence</th>
            </tr>
            {subject_rows}
        </table>
        
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
</body>
</html>
"""
        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SmartNotes: Intelligent Academic Note Organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF
  python smartnotes_cli.py --pdf document.pdf --output results/

  # Process batch with subject organization
  python smartnotes_cli.py --batch ./pdfs --output results/ --organize

  # Process with language model enabled
  python smartnotes_cli.py --batch ./pdfs --output results/ --use-lm

  # Process without organizing and generate HTML
  python smartnotes_cli.py --pdf document.pdf --output results/ --no-organize --html
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdf",
        type=str,
        help="Path to single PDF file to process"
    )
    input_group.add_argument(
        "--batch",
        type=str,
        help="Directory containing PDF files to process"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="File pattern for batch processing (default: *.pdf)"
    )
    
    # Processing options
    parser.add_argument(
        "--use-lm",
        action="store_true",
        default=True,
        help="Use language model for OCR (default: True)"
    )
    parser.add_argument(
        "--no-lm",
        action="store_true",
        help="Disable language model"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    
    # Organization options
    parser.add_argument(
        "--organize",
        action="store_true",
        default=True,
        help="Organize output by subject (default: True)"
    )
    parser.add_argument(
        "--no-organize",
        action="store_true",
        help="Don't organize output by subject"
    )
    
    # Reporting options
    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Generate summary report (default: True)"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML summary"
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Resolve language model
    use_lm = not args.no_lm if args.use_lm else args.use_lm
    
    # Resolve organization
    organize = not args.no_organize if args.organize else args.organize
    
    # Create app
    app = SmartNotesApp(use_lm=use_lm, device=args.device, verbose=args.verbose)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    app.logger.info("="*70)
    app.logger.info("SmartNotes - Intelligent Academic Note Organization")
    app.logger.info("="*70)
    
    # Process
    if args.pdf:
        # Single PDF
        result = app.process_single_pdf(args.pdf, args.output, organize=organize)
        if result and args.html:
            app.create_web_summary(args.output)
    
    elif args.batch:
        # Batch processing
        results = app.process_batch(
            args.batch,
            args.output,
            pattern=args.pattern,
            organize=organize,
            report=args.report
        )
        
        if results and args.html:
            app.create_web_summary(args.output)
        
        app.logger.info(f"Batch processing complete: {len(results)} PDFs processed")
    
    app.logger.info("="*70)
    app.logger.info(f"Results saved to: {output_dir}")
    app.logger.info("="*70)


if __name__ == "__main__":
    main()
