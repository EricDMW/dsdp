#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point for Wireless Communication Environment

This script serves as the main entry point for training the wireless communication environment.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import sys
import os
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Set research plotting style
try:
    from toolkit.plotkit import set_research_style
    set_research_style()
    print("Research plotting style applied")
except ImportError:
    print("Warning: toolkit.plotkit not available. Using default plotting style.")

from dsdp.wireless_comm.config import get_args, save_hyperparameters, print_hyperparameters
from dsdp.wireless_comm.trainer import WirelessTrainer


def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None, help='Run directory to use (created by manager)')
    args, unknown = parser.parse_known_args()
    try:
        # Get arguments with parameter tuning
        print("Loading parameters...")
        user_args = get_args()
        # Merge run-dir into user_args if provided
        if args.run_dir:
            user_args.run_dir = args.run_dir
        # Print hyperparameters
        print_hyperparameters(user_args)
        
        # Remove redundant hyperparameter saving here
        # print("\nSaving hyperparameters...")
        # hyperparams_path = save_hyperparameters(args, args.save_dir)
        # if hasattr(args, 'parameters_dir'):
        #     params_path = save_hyperparameters(args, args.parameters_dir)
        #     print(f"Parameters from GUI saved to: {params_path}")
        # print(f"Hyperparameters saved to: {hyperparams_path}")
        
        # Create and run trainer
        print("\nInitializing trainer...")
        trainer = WirelessTrainer(user_args)
        
        # Start training
        print("\nStarting training...")
        run_number = trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Run directory: {trainer.run_dir}")
        print(f"📂 Run number: {run_number}")
        print(f"\n🚀 To execute the trained model:")
        print(f"   python -m dsdp.wireless_comm.execute_model --run_path {trainer.run_dir}")
        print(f"\n📊 To run evaluation with animations:")
        print(f"   python -m dsdp.wireless_comm.execute_model --run_path {trainer.run_dir} --num_episodes 5")
        print(f"\n🎬 To run single episode:")
        print(f"   python -m dsdp.wireless_comm.execute_model --run_path {trainer.run_dir} --single_episode")
        print(f"\n📁 Run contents:")
        print(f"   📄 Parameters: {trainer.parameters_dir}")
        print(f"   📄 Hyperparameters: {trainer.hyperparameters_dir}")
        print(f"   🤖 Model: {trainer.model_dir}")
        print(f"   📈 Training Progress: {trainer.training_progress_dir}")
        print(f"   🎬 Execution: {trainer.execution_dir}")
        print(f"   ℹ️  Info: {trainer.info_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 