#!/usr/bin/env python3
"""
Quick setup script for integrating handwritten and printed notes into SmartNotes.

Run this script to:
1. Check dependencies
2. Extract PDFs to images
3. Prepare for training
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def check_dependency(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def check_command(cmd):
    """Check if a command line tool is available."""
    result = subprocess.run(['which', cmd], capture_output=True)
    return result.returncode == 0

def check_poppler():
    """Check if poppler is installed by trying to import pdf2image and test it."""
    try:
        import pdf2image
        # If we can import it, poppler should be available
        return True
    except Exception:
        return False

def main():
    print_header("SmartNotes: Handwritten & Printed Notes Integration Setup")
    
    # Step 1: Check Python dependencies
    print(f"{BLUE}Step 1: Checking Python dependencies...{RESET}")
    
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('cv2', 'cv2'),
        ('PIL', 'PIL'),
        ('pdf2image', 'pdf2image'),
    ]
    
    missing_packages = []
    for package, import_name in required_packages:
        if check_dependency(package, import_name):
            print_success(f"{package} is installed")
        else:
            print_error(f"{package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n{YELLOW}Install missing packages with:{RESET}")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    # Step 2: Check poppler installation
    print(f"\n{BLUE}Step 2: Checking poppler installation...{RESET}")
    
    if check_poppler():
        print_success("poppler is installed")
    else:
        print_error("poppler is NOT installed")
        print(f"\n{YELLOW}Install poppler with:{RESET}")
        
        import platform
        system = platform.system()
        
        if system == "Darwin":  # macOS
            print("  brew install poppler")
        elif system == "Linux":
            print("  sudo apt-get install poppler-utils")
        elif system == "Windows":
            print("  choco install poppler")
            print("  or download from: https://github.com/oschwartz10612/poppler-windows/releases")
        
        return False
    
    # Step 3: Check dataset directories
    print(f"\n{BLUE}Step 3: Checking dataset directories...{RESET}")
    
    base_dir = Path(__file__).parent
    datasets_dir = base_dir / "datasets"
    
    handwritten_dir = datasets_dir / "handwritten notes"
    printed_dir = datasets_dir / "printed notes"
    
    handwritten_extracted = datasets_dir / "handwritten_notes_extracted"
    printed_extracted = datasets_dir / "printed_notes_extracted"
    
    if handwritten_dir.exists():
        pdf_count = len(list(handwritten_dir.rglob("*.pdf")))
        print_success(f"Handwritten notes directory found ({pdf_count} PDFs)")
    else:
        print_warning(f"Handwritten notes directory not found: {handwritten_dir}")
    
    if printed_dir.exists():
        pdf_count = len(list(printed_dir.rglob("*.pdf")))
        print_success(f"Printed notes directory found ({pdf_count} PDFs)")
    else:
        print_warning(f"Printed notes directory not found: {printed_dir}")
    
    # Step 4: Extract PDFs
    print(f"\n{BLUE}Step 4: Extract PDFs to images?{RESET}")
    
    extract_handwritten = False
    extract_printed = False
    
    if handwritten_dir.exists() and not handwritten_extracted.exists():
        try:
            response = input("Extract handwritten notes? (y/n): ").strip().lower()
            extract_handwritten = response == 'y'
        except EOFError:
            extract_handwritten = False
    
    if printed_dir.exists() and not printed_extracted.exists():
        try:
            response = input("Extract printed notes? (y/n): ").strip().lower()
            extract_printed = response == 'y'
        except EOFError:
            extract_printed = False
    
    if extract_handwritten:
        print(f"\n{YELLOW}Extracting handwritten notes... (this may take a few minutes){RESET}")
        os.makedirs(handwritten_extracted, exist_ok=True)
        
        try:
            from src.dataloader.pdf_processor import process_notes_directory
            
            # Process each subfolder separately to avoid memory issues
            total = 0
            for subfolder in handwritten_dir.iterdir():
                if subfolder.is_dir():
                    print(f"{YELLOW}Processing {subfolder.name}...{RESET}")
                    subfolder_total = process_notes_directory(
                        str(subfolder),
                        str(handwritten_extracted),
                        dpi=150
                    )
                    total += subfolder_total
            
            print_success(f"Extracted {total} images from handwritten notes")
        except Exception as e:
            print_error(f"Failed to extract handwritten notes: {e}")
            return False
    
    if extract_printed:
        print(f"\n{YELLOW}Extracting printed notes... (this may take a few minutes){RESET}")
        os.makedirs(printed_extracted, exist_ok=True)
        
        try:
            from src.dataloader.pdf_processor import process_notes_directory
            
            # Process each subfolder separately to avoid memory issues
            total = 0
            for subfolder in printed_dir.iterdir():
                if subfolder.is_dir():
                    print(f"{YELLOW}Processing {subfolder.name}...{RESET}")
                    subfolder_total = process_notes_directory(
                        str(subfolder),
                        str(printed_extracted),
                        dpi=150
                    )
                    total += subfolder_total
            
            print_success(f"Extracted {total} images from printed notes")
        except Exception as e:
            print_error(f"Failed to extract printed notes: {e}")
            return False
    
    # Step 5: Verify extracted datasets
    print(f"\n{BLUE}Step 5: Verifying extracted datasets...{RESET}")
    
    if handwritten_extracted.exists():
        image_count = len(list(handwritten_extracted.rglob("*.png")))
        if image_count > 0:
            print_success(f"Handwritten notes: {image_count} images extracted")
        else:
            print_warning("No images found in handwritten_notes_extracted")
    
    if printed_extracted.exists():
        image_count = len(list(printed_extracted.rglob("*.png")))
        if image_count > 0:
            print_success(f"Printed notes: {image_count} images extracted")
        else:
            print_warning("No images found in printed_notes_extracted")
    
    # Final summary
    print_header("Setup Complete!")
    
    print(f"{GREEN}Your SmartNotes project is ready for training!{RESET}\n")
    
    print("Next steps:")
    print("1. Review NOTES_INTEGRATION_GUIDE.md for advanced options")
    print("2. Start training: python src/training/train_ocr.py")
    print("3. Monitor training: tail -f smartnotes.log")
    print("\nFor more information, see:")
    print("  - NOTES_INTEGRATION_GUIDE.md (detailed guide)")
    print("  - QUICKSTART.md (quick start guide)")
    print("  - README.md (project overview)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
