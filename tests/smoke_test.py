#!/usr/bin/env python
"""
Smoke test for CMVRA publication readiness.

This test verifies basic functionality without requiring large datasets or GPU resources.
"""

import sys
import torch


def test_imports():
    """Test that core modules can be imported."""
    try:
        # Test imports with proper path (VIP/src is where zeta module is)
        import sys
        import os
        vip_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VIP', 'src'))
        sys.path.insert(0, vip_src_path)
        
        from zeta.data_loader import load_dataloaders
        from zeta.model_init import initialize_vip_encoder
        print("✓ Core imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_pytorch_setup():
    """Test PyTorch setup."""
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    return True


def test_torch_seed():
    """Test that random seed can be set."""
    try:
        torch.manual_seed(42)
        print("✓ Random seed setting works")
        return True
    except Exception as e:
        print(f"✗ Seed setting failed: {e}")
        return False


def test_json_config():
    """Test that JSON config parsing works."""
    import json
    try:
        test_config = {
            "task": "test",
            "modalities": ["rgb", "depth"],
            "batch_size": 1
        }
        json_str = json.dumps(test_config)
        parsed = json.loads(json_str)
        assert parsed["task"] == "test"
        print("✓ JSON config parsing works")
        return True
    except Exception as e:
        print(f"✗ JSON test failed: {e}")
        return False


def test_basic_math():
    """Test basic tensor operations."""
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        assert y.shape == (3,)
        print("✓ Basic tensor operations work")
        return True
    except Exception as e:
        print(f"✗ Tensor test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("CMVRA Smoke Test")
    print("=" * 60)
    
    tests = [
        ("PyTorch Setup", test_pytorch_setup),
        ("Random Seed", test_torch_seed),
        ("JSON Config", test_json_config),
        ("Tensor Ops", test_basic_math),
        ("Core Imports", test_imports),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- Testing: {name} ---")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All smoke tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
