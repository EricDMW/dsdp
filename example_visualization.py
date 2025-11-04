#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using Visualization with Real Training Data

This script demonstrates how to use the visualization tools with actual training data.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
from visualization import TrainingVisualizer


def create_sample_training_log():
    """Create a sample training log for demonstration."""
    # Generate realistic training data
    episodes = np.arange(1000)
    
    # Simulate training progress
    rewards = 50 * (1 - np.exp(-episodes / 200)) + 20 * np.exp(-episodes / 500)
    losses = 100 * np.exp(-episodes / 150)
    violations = 10 * np.exp(-episodes / 100)
    
    # Add some noise
    rewards += np.random.normal(0, 2, len(episodes))
    losses += np.random.normal(0, 1, len(episodes))
    violations += np.random.normal(0, 0.5, len(episodes))
    
    # Create DataFrame
    df = pd.DataFrame({
        'episode': episodes,
        'cumulative_reward': rewards,
        'training_loss': losses,
        'constraint_violations': violations,
        'learning_rate': 0.0005 * (1 - episodes / episodes[-1])
    })
    
    # Save to CSV
    df.to_csv('training_log.csv', index=False)
    print("Sample training log created: training_log.csv")
    
    return df


def main():
    """Main example function."""
    print("=== DSDP Training Visualization Example ===\n")
    
    # Create sample training data
    print("1. Creating sample training log...")
    training_df = create_sample_training_log()
    
    # Initialize visualizer
    print("2. Initializing visualizer...")
    visualizer = TrainingVisualizer()
    
    # Create output directory
    output_dir = Path("example_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Basic training progress
    print("3. Creating basic training progress plot...")
    data = visualizer.generate_synthetic_training_data(timesteps=1000, num_runs=3)
    visualizer.plot_training_progress(data, 
                                    save_path=output_dir / "example_training_progress.png")
    
    # Example 2: Algorithm comparison
    print("4. Creating algorithm comparison plot...")
    visualizer.plot_algorithm_comparison(save_path=output_dir / "example_algorithm_comparison.png")
    
    # Example 3: Environment analysis
    print("5. Creating environment analysis plot...")
    visualizer.plot_environment_analysis(save_path=output_dir / "example_environment_analysis.png")
    
    # Example 4: Hyperparameter analysis
    print("6. Creating hyperparameter analysis plot...")
    visualizer.plot_hyperparameter_analysis(save_path=output_dir / "example_hyperparameter_analysis.png")
    
    # Example 5: Publication figure
    print("7. Creating publication-ready figure...")
    visualizer.create_publication_figure(save_path=output_dir / "example_publication_figure.png")
    
    print(f"\n✅ All example plots saved to: {output_dir}")
    print("📊 Plots are publication-ready with research-quality styling!")
    
    # Display summary statistics
    print("\n=== Training Summary ===")
    print(f"Final Reward: {training_df['cumulative_reward'].iloc[-1]:.2f}")
    print(f"Final Loss: {training_df['training_loss'].iloc[-1]:.2f}")
    print(f"Final Violations: {training_df['constraint_violations'].iloc[-1]:.2f}")
    print(f"Total Episodes: {len(training_df)}")
    
    # Show hyperparameter info
    hyperparams = visualizer.load_hyperparameters()
    if hyperparams:
        print(f"\n=== Hyperparameters ===")
        print(f"Grid Size: {hyperparams.get('grid_size', 'N/A')}")
        print(f"Actor LR: {hyperparams.get('actor_lr', 'N/A')}")
        print(f"Critic LR: {hyperparams.get('critic_lr', 'N/A')}")
        print(f"Gamma: {hyperparams.get('gamma', 'N/A')}")
        print(f"Hidden Size: {hyperparams.get('hidden_size', 'N/A')}")


if __name__ == "__main__":
    main() 