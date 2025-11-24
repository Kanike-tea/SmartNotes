#!/usr/bin/env python3
"""
Configuration validator and fixer for SmartNotes.
Ensures all modules use consistent checkpoint paths.

Usage:
    python config_fix.py
"""

import os
from pathlib import Path
from smartnotes.paths import setup_imports

setup_imports()


def find_available_checkpoint():
    """Find which checkpoint actually exists."""
    checkpoints_dir = Path("checkpoints")
    
    if not checkpoints_dir.exists():
        return None, "Checkpoints directory doesn't exist"
    
    # Priority order - most reliable first
    checkpoint_candidates = [
        "ocr_epoch_6.pth",
        "ocr_best.pth",
        "ocr_finetuned_stage2_best.pth",
        "ocr_finetuned_epoch_20.pth",
        "ocr_final.pth",
    ]
    
    found = []
    for ckpt in checkpoint_candidates:
        path = checkpoints_dir / ckpt
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            found.append((ckpt, size_mb))
    
    return found, None


def main():
    print("\n" + "=" * 70)
    print("SmartNotes Configuration Validator & Fixer")
    print("=" * 70)
    
    # Check for checkpoints
    checkpoints, error = find_available_checkpoint()
    
    if error:
        print(f"\n❌ {error}")
        print("\nPlease ensure you have a trained model in checkpoints/")
        print("Supported checkpoints:")
        print("  - ocr_epoch_6.pth")
        print("  - ocr_best.pth")
        print("  - ocr_finetuned_stage2_best.pth")
        print("  - ocr_finetuned_epoch_20.pth")
        print("\nTo train a model:")
        print("  python src/training/train_ocr.py")
        return False
    
    if not checkpoints:
        print("\n❌ No checkpoints found in checkpoints/")
        return False
    
    print("\n✓ Found available checkpoints:")
    for ckpt, size in checkpoints:
        print(f"  - {ckpt} ({size:.2f} MB)")
    
    # Recommend checkpoint (first in priority order)
    recommended = checkpoints[0][0]
    print(f"\n✓ Recommended: {recommended}")
    
    # Analyze current configuration
    print("\n" + "=" * 70)
    print("Current Configuration Analysis")
    print("=" * 70)
    
    current_checkpoints = set()
    
    # Check preprocessing/recognize.py
    try:
        recognize_path = Path("preprocessing/recognize.py")
        if recognize_path.exists():
            content = recognize_path.read_text()
            # Extract checkpoint reference
            for line in content.split('\n'):
                if 'checkpoint_path=' in line and '.pth' in line:
                    # Extract filename
                    import re
                    match = re.search(r'checkpoints/([^"\']+)', line)
                    if match:
                        current_checkpoints.add(match.group(1))
            print(f"✓ preprocessing/recognize.py:")
            if current_checkpoints:
                for ckpt in current_checkpoints:
                    status = "✓" if ckpt in [c[0] for c in checkpoints] else "❌"
                    print(f"  {status} {ckpt}")
            else:
                print(f"  ⚠️  Could not parse checkpoint reference")
    except Exception as e:
        print(f"  ⚠️  Error reading file: {e}")
    
    # Check config.py
    try:
        config_path = Path("config.py")
        if config_path.exists():
            content = config_path.read_text()
            for line in content.split('\n'):
                if 'CHECKPOINT_PATH' in line and '.pth' in line:
                    import re
                    match = re.search(r'["\']([\w_\.]+)["\']', line)
                    if match:
                        ckpt = match.group(1)
                        status = "✓" if ckpt in [c[0] for c in checkpoints] else "❌"
                        print(f"✓ config.py:")
                        print(f"  {status} {ckpt}")
    except Exception as e:
        print(f"  ⚠️  Error reading file: {e}")
    
    # Check demo_gradio_notes.py
    try:
        demo_path = Path("src/inference/demo_gradio_notes.py")
        if demo_path.exists():
            content = demo_path.read_text()
            if 'checkpoint_path=' in content:
                import re
                match = re.search(r'checkpoint_path="([^"]+)"', content)
                if match:
                    ckpt = match.group(1).split('/')[-1]
                    status = "✓" if ckpt in [c[0] for c in checkpoints] else "❌"
                    print(f"✓ src/inference/demo_gradio_notes.py:")
                    print(f"  {status} {ckpt}")
    except Exception as e:
        print(f"  ⚠️  Error reading file: {e}")
    
    # Update files to use recommended checkpoint
    print("\n" + "=" * 70)
    print("Updating Configuration Files")
    print("=" * 70)
    
    updates = [
        {
            "file": "preprocessing/recognize.py",
            "patterns": [
                ('checkpoint_path="checkpoints/ocr_finetuned_stage2_best.pth"',
                 f'checkpoint_path="checkpoints/{recommended}"'),
                ('checkpoint_path="checkpoints/ocr_epoch_6.pth"',
                 f'checkpoint_path="checkpoints/{recommended}"'),
            ]
        },
        {
            "file": "config.py",
            "patterns": [
                ('CHECKPOINT_PATH: str = str(CHECKPOINTS_DIR / "ocr_epoch_6.pth")',
                 f'CHECKPOINT_PATH: str = str(CHECKPOINTS_DIR / "{recommended}")'),
                ('CHECKPOINT_PATH: str = str(CHECKPOINTS_DIR / "ocr_finetuned_stage2_best.pth")',
                 f'CHECKPOINT_PATH: str = str(CHECKPOINTS_DIR / "{recommended}")'),
            ]
        },
        {
            "file": "src/inference/demo_gradio_notes.py",
            "patterns": [
                ('checkpoint_path="checkpoints/ocr_finetuned_stage2_best.pth"',
                 f'checkpoint_path="checkpoints/{recommended}"'),
                ('checkpoint_path="checkpoints/ocr_epoch_6.pth"',
                 f'checkpoint_path="checkpoints/{recommended}"'),
            ]
        }
    ]
    
    files_updated = 0
    total_updates = 0
    
    for update_config in updates:
        filepath = Path(update_config["file"])
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
        
        content = filepath.read_text()
        updated = False
        
        for old, new in update_config["patterns"]:
            if old in content:
                content = content.replace(old, new)
                updated = True
                total_updates += 1
        
        if updated:
            filepath.write_text(content)
            print(f"✓ Updated: {filepath}")
            files_updated += 1
        else:
            # Check if file already uses the recommended checkpoint
            if recommended in content:
                print(f"✓ Already correct: {filepath}")
            else:
                print(f"⚠️  Could not find update pattern in: {filepath}")
    
    print("\n" + "=" * 70)
    print("Configuration Update Complete!")
    print("=" * 70)
    print(f"\n✓ Files updated: {files_updated}")
    print(f"✓ Total changes: {total_updates}")
    print(f"✓ All modules now use: checkpoints/{recommended}")
    
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("\n1. Verify configuration:")
    print("   python test_model.py")
    print("\n2. Test OCR on an image:")
    print("   python quick_test.py path/to/image.png")
    print("\n3. Run detailed diagnostics:")
    print("   python diagnose_image.py path/to/image.png --debug")
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
