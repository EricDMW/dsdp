#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify loss recording setup.

This script performs basic checks to ensure the environment is correctly configured
for running the two-phase training with loss recording.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required imports work."""
    print("🔍 Testing imports...")
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_success = True
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"  ❌ {display_name} import failed: {e}")
            all_success = False
    
    # Test custom modules
    try:
        from dsdp.wireless_comm.trainer_with_loss_record import WirelessTrainerWithLossRecord
        print(f"  ✅ WirelessTrainerWithLossRecord imported successfully")
    except ImportError as e:
        print(f"  ❌ WirelessTrainerWithLossRecord import failed: {e}")
        all_success = False
    
    try:
        from dsdp.wireless_comm.config import get_args
        print(f"  ✅ config.get_args imported successfully")
    except ImportError as e:
        print(f"  ❌ config.get_args import failed: {e}")
        all_success = False
    
    return all_success


def test_cuda():
    """Test CUDA availability."""
    print("\n🔍 Testing CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA is available")
            print(f"     Device count: {torch.cuda.device_count()}")
            print(f"     Current device: {torch.cuda.current_device()}")
            print(f"     Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"  ⚠️  CUDA is not available (will use CPU)")
            return True
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        return False


def test_directory_structure():
    """Test that the directory structure is correct."""
    print("\n🔍 Testing directory structure...")
    
    module_dir = Path(__file__).parent
    required_dirs = []
    optional_dirs = ['runs']
    
    all_success = True
    for dir_name in required_dirs:
        dir_path = module_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ exists")
        else:
            print(f"  ❌ {dir_name}/ does not exist")
            all_success = False
    
    for dir_name in optional_dirs:
        dir_path = module_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ exists")
        else:
            print(f"  ℹ️  {dir_name}/ does not exist (will be created during training)")
    
    return all_success


def test_reference_run_loading():
    """Test reference run loading (if a run exists)."""
    print("\n🔍 Testing reference run loading...")
    
    module_dir = Path(__file__).parent
    runs_dir = module_dir / "runs"
    
    if not runs_dir.exists():
        print(f"  ℹ️  No runs directory found (expected for first run)")
        return True
    
    existing_runs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not existing_runs:
        print(f"  ℹ️  No existing runs found (expected for first run)")
        return True
    
    print(f"  ℹ️  Found {len(existing_runs)} existing run(s)")
    
    # Try to find a run with final_lagrangian.json
    valid_reference_runs = []
    for run_dir in existing_runs:
        lagrangian_file = run_dir / "info" / "final_lagrangian.json"
        if lagrangian_file.exists():
            valid_reference_runs.append(run_dir)
            print(f"  ✅ {run_dir.name} has final_lagrangian.json")
    
    if not valid_reference_runs:
        print(f"  ⚠️  No runs with final_lagrangian.json found")
        print(f"     Run Phase 1 first to create a reference run")
    else:
        print(f"  ✅ Found {len(valid_reference_runs)} valid reference run(s)")
        print(f"     You can use any of these for Phase 2:")
        for run_dir in valid_reference_runs:
            print(f"       - {run_dir}")
    
    return True


def test_minimal_training():
    """Test a minimal training setup (without actually training)."""
    print("\n🔍 Testing minimal training setup...")
    
    try:
        import argparse
        from dsdp.wireless_comm.config import get_args
        
        # Simulate minimal arguments
        original_argv = sys.argv.copy()
        sys.argv = [
            'test',
            '--total-timesteps', '100',
            '--grid-size', '3',
            '--seed', '42',
            '--tune-parameters', 'false'
        ]
        
        try:
            args = get_args()
            print(f"  ✅ Arguments parsed successfully")
            print(f"     Grid size: {args.grid_size}")
            print(f"     Total timesteps: {args.total_timesteps}")
            print(f"     Seed: {args.seed}")
            return True
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"  ❌ Error in minimal training setup: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("LOSS RECORDING SETUP TEST")
    print("="*70)
    print("\nThis script will verify that your environment is correctly set up")
    print("for running the two-phase training with loss recording.\n")
    
    results = {}
    
    results['imports'] = test_imports()
    results['cuda'] = test_cuda()
    results['directory'] = test_directory_structure()
    results['reference_loading'] = test_reference_run_loading()
    results['minimal_training'] = test_minimal_training()
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✅ All tests passed! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Run Phase 1 (reference training):")
        print("     python -m dsdp.wireless_comm.example_loss_record_training --phase 1 --total-timesteps 10000")
        print("  2. Run Phase 2 (recording training) with the reference run directory")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")
    
    print("")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

