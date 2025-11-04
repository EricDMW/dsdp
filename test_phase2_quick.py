#!/usr/bin/env python3
"""Quick test to verify Phase 2 loss recording works."""

import sys
from pathlib import Path

# Test Phase 2 with minimal timesteps
if __name__ == "__main__":
    print("="*70)
    print("QUICK PHASE 2 TEST")
    print("="*70)
    
    # Set minimal arguments
    sys.argv = [
        'test',
        '--phase', '2',
        '--reference-run-dir', './runs/run_128',
        '--total-timesteps', '100',
        '--grid-size', '3',
        '--seed', '999',
        '--tune-parameters', 'false'
    ]
    
    # Import and run
    from dsdp.wireless_comm.example_loss_record_training import main
    
    print("\nStarting Phase 2 test with 100 timesteps...")
    main()

