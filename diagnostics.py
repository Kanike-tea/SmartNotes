#!/usr/bin/env python3
"""
SmartNotes Advanced Diagnostics

Detailed troubleshooting tool that shows exactly what's happening
at each step of the OCR pipeline. Helps identify where issues occur.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import OrderedDict

from smartnotes.paths import setup_imports

setup_imports()


class DiagnosticsReport:
    def __init__(self, image_path, verbose=False):
        self.image_path = Path(image_path)
        self.verbose = verbose
        self.report = OrderedDict()
        self.start_time = time.time()
        self.step_times = {}
    
    def add_step(self, name, status, details=None, timing=None):
        """Add a step to the diagnostics report"""
        step = {
            'status': status,
            'details': details or {},
        }
        if timing:
            step['timing_ms'] = timing
        
        self.report[name] = step
    
    def export_json(self, output_path=None):
        """Export report as JSON"""
        if output_path is None:
            output_path = self.image_path.parent / f"diagnostics_{self.image_path.stem}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        return output_path
    
    def print_report(self):
        """Print formatted report"""
        print("\n" + "=" * 70)
        print("SmartNotes Diagnostics Report")
        print("=" * 70 + "\n")
        
        for step_name, step_data in self.report.items():
            status = step_data['status']
            status_symbol = '✓' if status == 'success' else '✗' if status == 'failed' else '⚠'
            
            print(f"{status_symbol} [{step_name}]")
            
            if step_data['details']:
                for key, value in step_data['details'].items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    - {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            
            if 'timing_ms' in step_data:
                print(f"  ⏱ Time: {step_data['timing_ms']:.1f}ms")
            
            print()


def run_diagnostics(image_path, verbose=False):
    """Run full diagnostics on an image"""
    
    report = DiagnosticsReport(image_path, verbose)
    
    print("\n" + "=" * 70)
    print("SmartNotes Advanced Diagnostics")
    print("=" * 70 + "\n")
    
    # Step 1: Validate image file
    print("[1/6] Validating image file...")
    step_start = time.time()
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        report.add_step('Image Validation', 'failed', {
            'error': f'File not found: {image_path}'
        }, (time.time() - step_start) * 1000)
        return report
    
    try:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"✗ Could not read image. Unsupported format?")
            report.add_step('Image Validation', 'failed', {
                'error': 'Could not decode image',
                'path': str(image_path),
                'file_size_kb': Path(image_path).stat().st_size / 1024
            }, (time.time() - step_start) * 1000)
            return report
        
        h, w = img.shape
        file_size_kb = Path(image_path).stat().st_size / 1024
        
        print(f"✓ Image loaded: {w}x{h}, {file_size_kb:.1f} KB")
        report.add_step('Image Validation', 'success', {
            'path': str(image_path),
            'dimensions': f'{w}x{h}',
            'file_size_kb': f'{file_size_kb:.1f}',
            'format': 'grayscale'
        }, (time.time() - step_start) * 1000)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        report.add_step('Image Validation', 'failed', {
            'error': str(e)
        }, (time.time() - step_start) * 1000)
        return report
    
    # Step 2: Load model
    print("[2/6] Loading OCR model...")
    step_start = time.time()
    
    try:
        from preprocessing.recognize import OCRRecognizer
        recognizer = OCRRecognizer()
        print(f"✓ Model loaded")
        report.add_step('Model Loading', 'success', {
            'model_class': 'OCRRecognizer',
            'device': str(recognizer.device) if hasattr(recognizer, 'device') else 'unknown'
        }, (time.time() - step_start) * 1000)
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        report.add_step('Model Loading', 'failed', {
            'error': str(e),
            'traceback': str(e)
        }, (time.time() - step_start) * 1000)
        return report
    
    # Step 3: Line segmentation
    print("[3/6] Segmenting lines...")
    step_start = time.time()
    
    try:
        from preprocessing.line_segment import segment_lines
        lines = segment_lines(image_path)
        
        if lines:
            print(f"✓ Found {len(lines)} lines")
            report.add_step('Line Segmentation', 'success', {
                'lines_found': len(lines),
                'line_count_details': f'{len(lines)} lines extracted'
            }, (time.time() - step_start) * 1000)
        else:
            print(f"⚠ No lines detected")
            report.add_step('Line Segmentation', 'warning', {
                'lines_found': 0,
                'message': 'Line detection returned empty list'
            }, (time.time() - step_start) * 1000)
            
    except Exception as e:
        print(f"✗ Line segmentation failed: {e}")
        report.add_step('Line Segmentation', 'failed', {
            'error': str(e)
        }, (time.time() - step_start) * 1000)
        lines = []
    
    # Step 4: Text recognition
    print("[4/6] Recognizing text...")
    step_start = time.time()
    text = None
    
    try:
        text = recognizer.predict(image_path)
        
        if text and not text.startswith("[NO TEXT"):
            line_count = len(text.split('\n'))
            char_count = len(text)
            print(f"✓ Recognition successful")
            print(f"  {line_count} lines, {char_count} characters")
            
            report.add_step('Text Recognition', 'success', {
                'lines_recognized': line_count,
                'total_characters': char_count,
                'preview': text[:100] + '...' if len(text) > 100 else text
            }, (time.time() - step_start) * 1000)
        else:
            print(f"⚠ No text recognized: {text}")
            report.add_step('Text Recognition', 'warning', {
                'result': text,
                'message': 'Recognition returned no text'
            }, (time.time() - step_start) * 1000)
            
    except Exception as e:
        print(f"✗ Recognition failed: {e}")
        report.add_step('Text Recognition', 'failed', {
            'error': str(e)
        }, (time.time() - step_start) * 1000)
    
    # Step 5: Quality metrics
    print("[5/6] Computing quality metrics...")
    step_start = time.time()
    
    try:
        metrics = {
            'image_quality': 'good' if h > 100 and w > 100 else 'low',
            'aspect_ratio': f'{w/h:.2f}',
            'total_pixels': h * w,
            'line_count': len(lines),
        }
        
        if 'text' in locals() and text and not text.startswith("[NO TEXT"):
            metrics['text_recognized'] = 'yes'
            metrics['recognition_length'] = len(text)
        else:
            metrics['text_recognized'] = 'no'
        
        print(f"✓ Metrics computed")
        report.add_step('Quality Metrics', 'success', metrics, (time.time() - step_start) * 1000)
        
    except Exception as e:
        print(f"✗ Metrics failed: {e}")
        report.add_step('Quality Metrics', 'failed', {
            'error': str(e)
        }, (time.time() - step_start) * 1000)
    
    # Step 6: Summary
    print("[6/6] Generating summary...")
    step_start = time.time()
    
    # Count successful steps
    successful = sum(1 for step in report.report.values() if step['status'] == 'success')
    total = len(report.report)
    
    summary = {
        'total_steps': total,
        'successful_steps': successful,
        'status': 'success' if successful == total else 'partial' if successful > 0 else 'failed',
        'total_time_ms': (time.time() - report.start_time) * 1000
    }
    
    report.add_step('Summary', summary['status'], summary, (time.time() - step_start) * 1000)
    
    # Print report
    print()
    report.print_report()
    
    # Export JSON
    json_path = report.export_json()
    print(f"📄 Detailed report saved to: {json_path}\n")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Advanced OCR diagnostics for troubleshooting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnostics.py --image test.png
  python diagnostics.py --image test.png --verbose
  python diagnostics.py --image test.png --json report.json
        """
    )
    
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1
    
    report = run_diagnostics(args.image, args.verbose)
    
    if args.json:
        json_path = report.export_json(args.json)
        print(f"Report saved to: {json_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
