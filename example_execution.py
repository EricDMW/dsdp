#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Model Execution Script

This script demonstrates how to execute trained models and generate animations.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Decentralized-Safe-MARL-with-General-Utilities'))

from execute_model import ModelExecutor, find_latest_run


def example_single_episode_execution():
    """Example: Execute a single episode and create animation."""
    print("🎬 Example: Single Episode Execution")
    print("=" * 50)
    
    # Get run path from user or use latest
    run_path = input("Enter path to run directory (e.g., 'run_1', '23', or full path) or press Enter for latest: ").strip()
    
    if not run_path:
        print("🔍 Using latest run...")
        latest_run = find_latest_run()
        if latest_run is None:
            print("❌ No run directories found. Please run training first.")
            return
        run_path = str(latest_run)
        print(f"📁 Using: {latest_run.name}")
    
    try:
        # Initialize executor
        executor = ModelExecutor(run_path=run_path, device="cpu")
        
        # Execute single episode
        frames, stats = executor.execute_episode(max_steps=50, render=True)
        
        # Create animation
        animation_path = executor.create_animation(frames, fps=3)
        
        print(f"\n✅ Single episode execution completed!")
        print(f"📊 Episode Statistics:")
        print(f"   Total Reward: {stats['total_reward']:.2f}")
        print(f"   Steps: {stats['steps']}")
        print(f"   Successful Transmissions: {stats['successful_transmissions']}")
        print(f"   Failed Transmissions: {stats['failed_transmissions']}")
        print(f"   Action Distribution: {stats['action_distribution']}")
        print(f"🎬 Animation saved to: {animation_path}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


def example_multi_episode_evaluation():
    """Example: Run multi-episode evaluation."""
    print("🔬 Example: Multi-Episode Evaluation")
    print("=" * 50)
    
    # Get run path from user or use latest
    run_path = input("Enter path to run directory (e.g., 'run_1', '23', or full path) or press Enter for latest: ").strip()
    
    if not run_path:
        print("🔍 Using latest run...")
        latest_run = find_latest_run()
        if latest_run is None:
            print("❌ No run directories found. Please run training first.")
            return
        run_path = str(latest_run)
        print(f"📁 Using: {latest_run.name}")
    
    try:
        # Initialize executor
        executor = ModelExecutor(run_path=run_path, device="cpu")
        
        # Run evaluation
        results = executor.run_evaluation(
            num_episodes=3,
            max_steps=50,
            fps=3
        )
        
        print(f"\n✅ Multi-episode evaluation completed!")
        print(f"📊 Evaluation Results:")
        print(f"   Run Number: {results['run_number']}")
        print(f"   Model ID: {results['model_id']}")
        print(f"   Episodes: {results['completed_episodes']}/{results['num_episodes']}")
        print(f"   Average Reward: {results['average_reward']:.2f}")
        print(f"   Average Steps: {results['average_steps']:.1f}")
        print(f"   Average Successful Transmissions: {results['average_successful_transmissions']:.1f}")
        print(f"   Average Failed Transmissions: {results['average_failed_transmissions']:.1f}")
        print(f"🎬 Animations created: {len(results['animation_paths'])}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def example_batch_execution():
    """Example: Execute multiple runs in batch."""
    print("📦 Example: Batch Run Execution")
    print("=" * 50)
    
    # Get directory containing multiple run folders
    runs_dir = input("Enter path to runs directory (e.g., 'dsdp/wireless_comm/runs') or press Enter for default: ").strip()
    
    if not runs_dir:
        runs_dir = Path(__file__).parent / "runs"
    else:
        runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        print(f"❌ Directory not found: {runs_dir}")
        return
    
    # Find all run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        print(f"❌ No valid run directories found in {runs_dir}")
        return
    
    print(f"Found {len(run_dirs)} run directories:")
    for i, run_dir in enumerate(run_dirs):
        print(f"  {i+1}. {run_dir.name}")
    
    # Execute each run
    for i, run_dir in enumerate(run_dirs):
        print(f"\n--- Executing Run {i+1}/{len(run_dirs)}: {run_dir.name} ---")
        
        try:
            executor = ModelExecutor(run_path=str(run_dir), device="cpu")
            
            # Run single episode for each run
            frames, stats = executor.execute_episode(max_steps=30, render=True)
            animation_path = executor.create_animation(frames, fps=3)
            
            print(f"✅ Run {run_dir.name} executed successfully!")
            print(f"   Reward: {stats['total_reward']:.2f}")
            print(f"   Steps: {stats['steps']}")
            print(f"   Animation: {animation_path}")
            
        except Exception as e:
            print(f"❌ Error executing run {run_dir.name}: {e}")


def example_latest_run():
    """Example: Execute the latest run automatically."""
    print("🚀 Example: Latest Run Execution")
    print("=" * 50)
    
    # Find latest run
    latest_run = find_latest_run()
    if latest_run is None:
        print("❌ No run directories found!")
        print("   Please run training first:")
        print("   python -m dsdp.wireless_comm.main --total-timesteps 1000")
        return
    
    print(f"📁 Found latest run: {latest_run.name}")
    
    try:
        # Initialize executor
        executor = ModelExecutor(run_path=str(latest_run), device="cpu")
        
        # Execute single episode
        frames, stats = executor.execute_episode(max_steps=50, render=True)
        animation_path = executor.create_animation(frames, fps=3)
        
        print(f"\n✅ Latest run execution completed!")
        print(f"📊 Episode Statistics:")
        print(f"   Total Reward: {stats['total_reward']:.2f}")
        print(f"   Steps: {stats['steps']}")
        print(f"   Successful Transmissions: {stats['successful_transmissions']}")
        print(f"   Failed Transmissions: {stats['failed_transmissions']}")
        print(f"   Action Distribution: {stats['action_distribution']}")
        print(f"🎬 Animation saved to: {animation_path}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


def example_custom_execution():
    """Example: Execute with custom parameters."""
    print("⚙️  Example: Custom Execution Parameters")
    print("=" * 50)
    
    # Get run path
    run_path = input("Enter run path (or press Enter for latest): ").strip()
    
    if not run_path:
        latest_run = find_latest_run()
        if latest_run is None:
            print("❌ No run directories found.")
            return
        run_path = str(latest_run)
        print(f"📁 Using: {latest_run.name}")
    
    # Get custom parameters
    max_steps = input("Enter max steps (default 50): ").strip()
    max_steps = int(max_steps) if max_steps.isdigit() else 50
    
    fps = input("Enter FPS for animation (default 3): ").strip()
    fps = int(fps) if fps.isdigit() else 3
    
    num_episodes = input("Enter number of episodes (default 1): ").strip()
    num_episodes = int(num_episodes) if num_episodes.isdigit() else 1
    
    try:
        # Initialize executor
        executor = ModelExecutor(run_path=run_path, device="cpu")
        
        if num_episodes == 1:
            # Single episode
            frames, stats = executor.execute_episode(max_steps=max_steps, render=True)
            animation_path = executor.create_animation(frames, fps=fps)
            
            print(f"\n✅ Custom execution completed!")
            print(f"📊 Results:")
            print(f"   Steps: {stats['steps']}")
            print(f"   Reward: {stats['total_reward']:.2f}")
            print(f"   Animation: {animation_path}")
        else:
            # Multiple episodes
            results = executor.run_evaluation(
                num_episodes=num_episodes,
                max_steps=max_steps,
                fps=fps
            )
            
            print(f"\n✅ Custom evaluation completed!")
            print(f"📊 Results:")
            print(f"   Episodes: {results['completed_episodes']}/{results['num_episodes']}")
            print(f"   Average Reward: {results['average_reward']:.2f}")
            print(f"   Animations: {len(results['animation_paths'])}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


def example_run_comparison():
    """Example: Compare multiple runs."""
    print("📊 Example: Run Comparison")
    print("=" * 50)
    
    # Get run paths
    run_paths = []
    while True:
        run_path = input("Enter run path (or 'done' to finish): ").strip()
        if run_path.lower() == 'done':
            break
        if run_path:
            run_paths.append(run_path)
    
    if len(run_paths) < 2:
        print("❌ Need at least 2 runs for comparison.")
        return
    
    print(f"\nComparing {len(run_paths)} runs...")
    
    results = []
    for i, run_path in enumerate(run_paths):
        print(f"\n--- Run {i+1}: {run_path} ---")
        
        try:
            executor = ModelExecutor(run_path=run_path, device="cpu")
            
            # Execute single episode
            frames, stats = executor.execute_episode(max_steps=30, render=False)
            
            results.append({
                'run_path': run_path,
                'run_number': executor.metadata['run_number'],
                'model_id': executor.metadata['model_id'],
                'total_reward': stats['total_reward'],
                'steps': stats['steps'],
                'successful_transmissions': stats['successful_transmissions'],
                'failed_transmissions': stats['failed_transmissions'],
                'average_reward': stats['total_reward'] / max(1, stats['steps'])
            })
            
            print(f"✅ Completed: Reward={stats['total_reward']:.2f}, Steps={stats['steps']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'run_path': run_path,
                'error': str(e)
            })
    
    # Display comparison
    print(f"\n📊 Comparison Results:")
    print("-" * 80)
    print(f"{'Run':<15} {'Reward':<10} {'Steps':<8} {'Success':<8} {'Failed':<8} {'Avg Reward':<12}")
    print("-" * 80)
    
    for result in results:
        if 'error' in result:
            print(f"{result['run_path']:<15} {'ERROR':<10}")
        else:
            print(f"{result['run_path']:<15} {result['total_reward']:<10.2f} {result['steps']:<8} "
                  f"{result['successful_transmissions']:<8} {result['failed_transmissions']:<8} "
                  f"{result['average_reward']:<12.2f}")


def main():
    """Main function to run examples."""
    print("🚀 DSDP Wireless Communication Model Execution Examples")
    print("=" * 60)
    print()
    
    while True:
        print("Choose an example to run:")
        print("1. Single Episode Execution")
        print("2. Multi-Episode Evaluation")
        print("3. Batch Model Execution")
        print("4. Latest Run Execution")
        print("5. Custom Execution Parameters")
        print("6. Run Comparison")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            example_single_episode_execution()
        elif choice == "2":
            example_multi_episode_evaluation()
        elif choice == "3":
            example_batch_execution()
        elif choice == "4":
            example_latest_run()
        elif choice == "5":
            example_custom_execution()
        elif choice == "6":
            example_run_comparison()
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-7.")
        
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main() 