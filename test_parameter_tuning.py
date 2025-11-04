#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Parameter Tuning Functionality

This script tests the ParameterTuner functionality without argument conflicts.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import argparse
from pathlib import Path
import json

def test_parameter_tuning():
    """Test the ParameterTuner for hyperparameter tuning."""
    print("=== Parameter Tuning Test ===")
    
    try:
        from toolkit.parakit import ParameterTuner
        print("ParameterTuner available - launching GUI...")
        
        # Create a simple argument parser for demonstration
        parser = argparse.ArgumentParser(description="Parameter Tuning Demo")
        parser.add_argument("--grid_size", type=int, default=5, help="Grid size for environment")
        parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer size")
        parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
        
        # Create parameters directory
        params_dir = Path("test_parameters")
        params_dir.mkdir(exist_ok=True)
        
        # Configure ParameterTuner with validation
        tuner = ParameterTuner(
            parser=parser,
            save_path=str(params_dir),
            save_delay=10000,  # 10 seconds auto-save for demo
            auto_save_enabled=True,
            inactivity_timeout=10,  # 10 seconds inactivity timeout for demo
            validation_callbacks={
                'grid_size': lambda x: 2 <= int(x) <= 10,
                'learning_rate': lambda x: 1e-6 <= float(x) <= 1e-1,
                'batch_size': lambda x: 1 <= int(x) <= 128,
                'epochs': lambda x: 1 <= int(x) <= 1000,
                'hidden_size': lambda x: 16 <= int(x) <= 1024,
                'dropout': lambda x: 0.0 <= float(x) <= 0.9
            }
        )
        
        # Launch parameter tuning GUI
        print("GUI will open for parameter tuning...")
        print("You can modify parameters and they will be auto-saved.")
        print("The GUI will close automatically after 10 seconds of inactivity.")
        
        updated_parser = tuner.tune()
        args = updated_parser.parse_args([])  # Parse empty args to avoid conflicts
        
        print("\nFinal parameters from GUI:")
        print(f"  Grid Size: {args.grid_size}")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Hidden Size: {args.hidden_size}")
        print(f"  Dropout: {args.dropout}")
        
        # Check if parameters were saved
        param_files = list(params_dir.glob("*.json"))
        if param_files:
            latest_file = max(param_files, key=lambda x: x.stat().st_mtime)
            print(f"\nParameters saved to: {latest_file}")
            
            # Load and display saved parameters
            with open(latest_file, 'r') as f:
                saved_params = json.load(f)
            print(f"Saved parameters: {saved_params}")
        
        print("Parameter tuning test completed!")
        
    except ImportError:
        print("ParameterTuner not available. Skipping parameter tuning test.")
    except Exception as e:
        print(f"Parameter tuning failed: {e}")


if __name__ == "__main__":
    test_parameter_tuning() 