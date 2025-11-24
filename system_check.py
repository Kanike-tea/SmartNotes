#!/usr/bin/env python3
"""
SmartNotes System Check

Validates all dependencies, model files, and data paths are set up correctly.
Provides a complete setup status report before running OCR pipeline.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

from smartnotes.paths import setup_imports, get_project_root, get_checkpoint_dir, get_dataset_dir

setup_imports()


class SystemChecker:
    def __init__(self):
        self.base_path = get_project_root()
        self.results = defaultdict(dict)
        self.colors = {
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'end': '\033[0m'
        }
    
    def color(self, text, color='green'):
        return f"{self.colors[color]}{text}{self.colors['end']}"
    
    def check_python_version(self):
        """Check Python version"""
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        required = "3.8+"
        
        result = sys.version_info >= (3, 8)
        self.results['Python'][version] = result
        
        status = self.color("✓", "green") if result else self.color("✗", "red")
        print(f"{status} Python {version} (required: {required})")
        
        return result
    
    def check_imports(self):
        """Check critical imports"""
        imports = {
            'torch': 'PyTorch',
            'cv2': 'OpenCV',
            'numpy': 'NumPy',
            'gradio': 'Gradio',
        }
        
        for module, name in imports.items():
            try:
                __import__(module)
                self.results['Dependencies'][name] = True
                print(f"{self.color('✓', 'green')} {name}")
            except ImportError:
                self.results['Dependencies'][name] = False
                print(f"{self.color('✗', 'red')} {name} (MISSING)")
    
    def check_model_files(self):
        """Check model checkpoint files"""
        checkpoints_dir = self.base_path / 'checkpoints'
        expected_models = [
            'ocr_best.pth',
            'ocr_final.pth',
        ]
        
        if not checkpoints_dir.exists():
            print(f"{self.color('✗', 'red')} Checkpoints directory not found")
            self.results['Models']['directory'] = False
            return False
        
        for model in expected_models:
            path = checkpoints_dir / model
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.results['Models'][model] = True
                print(f"{self.color('✓', 'green')} {model} ({size_mb:.1f} MB)")
            else:
                self.results['Models'][model] = False
                print(f"{self.color('✗', 'red')} {model} (NOT FOUND)")
        
        return True
    
    def check_datasets(self):
        """Check dataset files"""
        datasets_dir = self.base_path / 'datasets'
        datasets = {
            'GNHK': 'Ground Truth Handwriting',
            'CensusHWR': 'Census Handwriting',
            'IAM': 'IAM Handwriting Database',
        }
        
        if not datasets_dir.exists():
            print(f"{self.color('✗', 'red')} Datasets directory not found")
            return False
        
        for dataset, desc in datasets.items():
            path = datasets_dir / dataset
            if path.exists():
                self.results['Datasets'][dataset] = True
                print(f"{self.color('✓', 'green')} {dataset} - {desc}")
            else:
                self.results['Datasets'][dataset] = False
                print(f"{self.color('⊘', 'yellow')} {dataset} - {desc} (not found)")
        
        return True
    
    def check_source_modules(self):
        """Check source modules"""
        src_dir = self.base_path / 'src'
        modules = {
            'model/ocr_model.py': 'OCR Model',
            'training/train_ocr.py': 'Training Script',
            'inference/test_ocr.py': 'Inference Test',
            'inference/demo_gradio.py': 'Gradio Demo',
            'dataloader/ocr_dataloader.py': 'Data Loader',
        }
        
        for module_path, desc in modules.items():
            path = src_dir / module_path
            if path.exists():
                self.results['Source Modules'][desc] = True
                print(f"{self.color('✓', 'green')} {desc}")
            else:
                self.results['Source Modules'][desc] = False
                print(f"{self.color('✗', 'red')} {desc} (NOT FOUND)")
    
    def check_preprocessing(self):
        """Check preprocessing modules"""
        preprocessing_dir = self.base_path / 'preprocessing'
        modules = {
            'line_segment.py': 'Line Segmentation',
            'recognize.py': 'Recognition',
            'pipeline.py': 'Pipeline',
            'postprocess.py': 'Post-processing',
            'text_preprocess.py': 'Text Preprocessing',
        }
        
        for module, desc in modules.items():
            path = preprocessing_dir / module
            if path.exists():
                self.results['Preprocessing'][desc] = True
                print(f"{self.color('✓', 'green')} {desc}")
            else:
                self.results['Preprocessing'][desc] = False
                print(f"{self.color('✗', 'red')} {desc} (NOT FOUND)")
    
    def run_all_checks(self):
        """Run all system checks"""
        print(self.color("\n" + "=" * 70, "blue"))
        print(self.color("SmartNotes System Check", "blue"))
        print(self.color("=" * 70 + "\n", "blue"))
        
        # Python version
        print(self.color("[1/6] Python Version", "blue"))
        self.check_python_version()
        print()
        
        # Dependencies
        print(self.color("[2/6] Dependencies", "blue"))
        self.check_imports()
        print()
        
        # Model files
        print(self.color("[3/6] Model Checkpoints", "blue"))
        self.check_model_files()
        print()
        
        # Datasets
        print(self.color("[4/6] Datasets", "blue"))
        self.check_datasets()
        print()
        
        # Source modules
        print(self.color("[5/6] Source Modules", "blue"))
        self.check_source_modules()
        print()
        
        # Preprocessing
        print(self.color("[6/6] Preprocessing Modules", "blue"))
        self.check_preprocessing()
        print()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary report"""
        total_checks = 0
        passed_checks = 0
        
        for category, checks in self.results.items():
            for name, result in checks.items():
                total_checks += 1
                if result:
                    passed_checks += 1
        
        print(self.color("=" * 70, "blue"))
        print(self.color("SUMMARY", "blue"))
        print(self.color("=" * 70, "blue"))
        
        pct = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        if passed_checks == total_checks:
            status = self.color("✓ ALL SYSTEMS GO", "green")
        elif pct >= 80:
            status = self.color("⚠ PARTIALLY OK", "yellow")
        else:
            status = self.color("✗ ISSUES FOUND", "red")
        
        print(f"\n{status}")
        print(f"Checks passed: {passed_checks}/{total_checks} ({pct:.0f}%)\n")
        
        # Detailed breakdown
        for category in sorted(self.results.keys()):
            checks = self.results[category]
            category_pass = sum(1 for v in checks.values() if v)
            category_total = len(checks)
            
            symbol = self.color("✓", "green") if category_pass == category_total else self.color("⚠", "yellow")
            print(f"{symbol} {category}: {category_pass}/{category_total}")
        
        print(f"\n{self.color('=' * 70, 'blue')}\n")
        
        return passed_checks == total_checks


def main():
    checker = SystemChecker()
    success = checker.run_all_checks()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
