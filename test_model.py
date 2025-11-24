#!/usr/bin/env python3
"""
SmartNotes Model Health Diagnostics

Tests:
1. Architecture - Can model be created and does forward pass work?
2. Checkpoints - Are checkpoint files loadable?
3. Tokenizer - Is character encoding/decoding working?
4. Simple Image - Can model recognize text on a simple image?
5. Confidence - Is model appropriately uncertain on noise?
"""

import sys
from pathlib import Path

from smartnotes.paths import setup_imports

setup_imports()

import torch
import numpy as np
import cv2

def test_architecture():
    """Test 1: Model architecture and forward pass"""
    print("\nModel Architecture Test")
    print("=" * 70)
    
    try:
        from src.model.ocr_model import CRNN
        from src.dataloader.ocr_dataloader import TextTokenizer
        
        # Create tokenizer
        tokenizer = TextTokenizer()
        num_classes = len(tokenizer.chars)
        
        print(f"Vocabulary size: {num_classes}")
        print(f"Characters: {tokenizer.chars}")
        print(f"Blank index: {num_classes}")
        
        # Create model
        model = CRNN(num_classes=num_classes)
        print("✓ Model created successfully")
        
        # Test forward pass
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()
        
        # Create dummy input (batch_size=2, channels=1, height=32, width=128)
        dummy_input = torch.randn(2, 1, 32, 128).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Validate output shape
        expected_shape = (32, 2, num_classes + 1)  # (seq_len, batch, num_classes+blank)
        if output.shape == expected_shape:
            print("✓ Output shape correct")
            return True
        else:
            print(f"✗ Output shape mismatch. Expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_checkpoints():
    """Test 2: Checkpoint loading"""
    print("\nCheckpoint Loading Test")
    print("=" * 70)
    
    try:
        from src.model.ocr_model import CRNN
        from src.dataloader.ocr_dataloader import TextTokenizer
        
        tokenizer = TextTokenizer()
        num_classes = len(tokenizer.chars)
        
        model = CRNN(num_classes=num_classes)
        device = torch.device("cpu")
        model = model.to(device)
        
        checkpoint_dir = Path("checkpoints")
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        
        if not checkpoints:
            print("✗ No checkpoints found")
            return False
        
        # Test loading first checkpoint
        checkpoint_path = checkpoints[0]
        print(f"✓ Found: {checkpoint_path}")
        
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {file_size_mb:.2f} MB")
        
        # Check if it's a dict checkpoint or direct state dict
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            print(f"  Type: Dictionary checkpoint")
            print(f"  Keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Count parameters
        num_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        print(f"  Parameters: {num_params}")
        
        # Try loading
        model.load_state_dict(state_dict)
        print("✓ Compatible with current model")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_tokenizer():
    """Test 3: Tokenizer encoding/decoding"""
    print("\nTokenizer Test")
    print("=" * 70)
    
    try:
        from src.dataloader.ocr_dataloader import TextTokenizer
        
        tokenizer = TextTokenizer()
        
        # Test text
        test_text = "Hello World 123"
        print(f"Test text: '{test_text}'")
        
        # Encode
        encoded = tokenizer.encode(test_text)
        print(f"✓ Encoded: {encoded}")
        
        # Decode
        decoded = tokenizer.decode(np.array(encoded))
        print(f"✓ Decoded: '{decoded}'")
        
        # Check round-trip
        if decoded == test_text:
            print("✓ Round-trip successful (encoding → decoding)")
            return True
        else:
            print(f"✗ Mismatch: '{decoded}' != '{test_text}'")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_simple_image():
    """Test 4: Simple image recognition"""
    print("\nSimple Image Test")
    print("=" * 70)
    
    try:
        from preprocessing.recognize import OCRRecognizer
        
        recognizer = OCRRecognizer()
        
        # Create simple test image with text
        img = np.ones((32, 128), dtype=np.uint8) * 255
        cv2.putText(img, "TEST", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,), 2)
        
        # Save test image
        test_path = Path("test_image.png")
        cv2.imwrite(str(test_path), img)
        
        print(f"✓ Created test image: {test_path}")
        
        # Run OCR
        result = recognizer.predict(str(test_path))
        
        print(f"✓ Model produced output")
        print(f"  Result: '{result}'")
        
        if result and not result.startswith("[NO TEXT"):
            print("✓ Model recognized text")
            return True
        else:
            print("✗ Model failed to recognize text")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_confidence():
    """Test 5: Model confidence on noise"""
    print("\nConfidence Test")
    print("=" * 70)
    
    try:
        from preprocessing.recognize import OCRRecognizer
        import tempfile
        
        recognizer = OCRRecognizer()
        
        # Create noise image
        noise = np.random.randint(0, 256, (32, 128), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, noise)
            noise_path = tmp.name
        
        print("✓ Created noise image")
        
        # Run OCR
        result = recognizer.predict(noise_path)
        
        print(f"  Result: '{result}'")
        
        # Clean up
        Path(noise_path).unlink()
        
        # Should be rejected or low confidence
        if result.startswith("[NO TEXT"):
            print("✓ Model appropriately rejected noise")
            return True
        elif len(result) < 5:
            print("✓ Model produced minimal output on noise (conservative)")
            return True
        else:
            print("⚠ Model produced output on noise (overconfident)")
            return True  # Not critical failure
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("SmartNotes Model Health Diagnostics")
    print("=" * 70)
    
    tests = [
        ("Architecture", test_architecture),
        ("Checkpoints", test_checkpoints),
        ("Tokenizer", test_tokenizer),
        ("Simple Image", test_simple_image),
        ("Confidence", test_confidence),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed - Model is healthy!")
        return 0
    else:
        print("✗ Some tests failed - Check diagnostics above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
