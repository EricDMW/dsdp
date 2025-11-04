#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for two-phase training with loss recording.

This script demonstrates how to:
1. Phase 1: Run reference training to establish baseline
2. Phase 2: Run training with loss recording to track convergence

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import argparse
from pathlib import Path

from .config import get_args
from .trainer_with_loss_record import train_with_loss_record
import sys


def main():
    """Main function for two-phase training."""
    
    # Check for phase and reference-run-dir in sys.argv
    phase = None
    reference_run_dir = None
    
    if '--phase' in sys.argv:
        phase_idx = sys.argv.index('--phase')
        if phase_idx + 1 < len(sys.argv):
            phase = int(sys.argv[phase_idx + 1])
            sys.argv.pop(phase_idx + 1)
            sys.argv.pop(phase_idx)
    
    if '--reference-run-dir' in sys.argv:
        ref_idx = sys.argv.index('--reference-run-dir')
        if ref_idx + 1 < len(sys.argv):
            reference_run_dir = sys.argv[ref_idx + 1]
            sys.argv.pop(ref_idx + 1)
            sys.argv.pop(ref_idx)
    
    # Validate phase
    if phase is None:
        print("Error: --phase is required. Use --phase 1 or --phase 2")
        print("\nUsage:")
        print("  Phase 1: python -m dsdp.wireless_comm.example_loss_record_training --phase 1 [other args]")
        print("  Phase 2: python -m dsdp.wireless_comm.example_loss_record_training --phase 2 --reference-run-dir PATH [other args]")
        sys.exit(1)
    
    if phase not in [1, 2]:
        print(f"Error: phase must be 1 or 2, got {phase}")
        sys.exit(1)
    
    # Parse remaining arguments using the standard config parser
    # Disable parameter tuning GUI for this automated workflow
    sys.argv.append('--tune-parameters')
    sys.argv.append('false')
    args = get_args()
    
    # Validate arguments
    if phase == 2 and reference_run_dir is None:
        print("Error: --reference-run-dir is required for phase 2")
        sys.exit(1)
    
    if phase == 1 and reference_run_dir is not None:
        print("⚠️  Warning: --reference-run-dir is ignored in phase 1")
        reference_run_dir = None
    
    # Run appropriate phase
    if phase == 1:
        print("="*70)
        print("PHASE 1: REFERENCE TRAINING")
        print("="*70)
        print("This phase will train the model and save:")
        print("  - Final Lagrangian multipliers")
        print("  - Final policy networks")
        print("="*70)
        
        run_number = train_with_loss_record(args, reference_run_dir=None)
        
        print("\n" + "="*70)
        print("PHASE 1 COMPLETED!")
        print("="*70)
        print(f"Run number: {run_number}")
        print(f"Run directory: ./dsdp/wireless_comm/runs/run_{run_number}")
        print("\nTo start Phase 2 (loss recording), run:")
        print(f"  python -m dsdp.wireless_comm.example_loss_record_training \\")
        print(f"    --phase 2 \\")
        print(f"    --reference-run-dir ./dsdp/wireless_comm/runs/run_{run_number} \\")
        print(f"    --total-timesteps {args.total_timesteps}")
        print("="*70)
        
    else:  # phase == 2
        print("="*70)
        print("PHASE 2: TRAINING WITH LOSS RECORDING")
        print("="*70)
        print(f"Reference directory: {reference_run_dir}")
        print("This phase will:")
        print("  - Train the model from scratch")
        print("  - Track Lagrangian error (1-norm)")
        print("  - Track policy convergence (cosine similarity)")
        print("  - Save results to lagrangian.csv")
        print("="*70)
        
        # Verify reference directory exists
        ref_path = Path(reference_run_dir)
        if not ref_path.exists():
            print(f"\n❌ Error: Reference directory not found: {ref_path}")
            return
        
        if not (ref_path / "info" / "final_lagrangian.json").exists():
            print(f"\n❌ Error: Reference Lagrangian file not found in {ref_path}")
            print("   Make sure Phase 1 completed successfully")
            return
        
        run_number = train_with_loss_record(args, reference_run_dir=str(ref_path))
        
        print("\n" + "="*70)
        print("PHASE 2 COMPLETED!")
        print("="*70)
        print(f"Run number: {run_number}")
        print(f"Run directory: ./dsdp/wireless_comm/runs/run_{run_number}")
        print("\nGenerated files:")
        print("  - lagrangian.csv (Lagrangian errors and policy similarities)")
        print("  - lagrangian_convergence_*.png (Convergence plots)")
        print("="*70)


if __name__ == "__main__":
    main()

