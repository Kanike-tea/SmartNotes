#!/usr/bin/env python3
"""
Standalone script to extract printed notes with folder-by-folder processing.
This avoids memory issues by processing one subject folder at a time.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    base_dir = Path(__file__).parent
    datasets_dir = base_dir / "datasets"
    printed_dir = datasets_dir / "printed notes"
    printed_extracted = datasets_dir / "printed_notes_extracted"
    
    print("=" * 70)
    print("SmartNotes: Printed Notes Extraction (Folder-by-Folder)")
    print("=" * 70)
    
    if not printed_dir.exists():
        print(f"❌ Printed notes directory not found: {printed_dir}")
        return False
    
    # Count PDFs
    pdf_files = list(printed_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {printed_dir}")
        return False
    
    print(f"✓ Found {len(pdf_files)} PDF files")
    
    # Create output directory
    os.makedirs(printed_extracted, exist_ok=True)
    print(f"✓ Output directory: {printed_extracted}")
    
    # Import processor
    try:
        from src.dataloader.pdf_processor import process_notes_directory
    except ImportError as e:
        print(f"❌ Failed to import pdf_processor: {e}")
        return False
    
    # Process each subject folder
    total_images = 0
    folders = [f for f in printed_dir.iterdir() if f.is_dir()]
    
    print(f"\n📂 Found {len(folders)} subject folders:")
    for folder in sorted(folders):
        print(f"   - {folder.name}")
    
    print(f"\n⏳ Processing folders...")
    for i, subfolder in enumerate(sorted(folders), 1):
        try:
            print(f"\n[{i}/{len(folders)}] Processing {subfolder.name}...", end=" ", flush=True)
            count = process_notes_directory(
                str(subfolder),
                str(printed_extracted),
                dpi=150
            )
            total_images += count
            print(f"✓ {count} images")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print(f"✓ Extraction Complete!")
    print(f"  Total images extracted: {total_images}")
    print(f"  Output directory: {printed_extracted}")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
