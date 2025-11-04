#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for Execution System

This script tests the robust execution system with various scenarios.

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


def test_latest_run_detection():
    """Test automatic latest run detection."""
    print("🔍 Testing latest run detection...")
    
    latest_run = find_latest_run()
    if latest_run is None:
        print("❌ No runs found. Please run training first.")
        return False
    
    print(f"✅ Latest run detected: {latest_run.name}")
    return True


def test_executor_initialization():
    """Test executor initialization with latest run."""
    print("\n🚀 Testing executor initialization...")
    
    try:
        executor = ModelExecutor(auto_find_latest=True)
        print(f"✅ Executor initialized successfully for: {executor.run_path.name}")
        print(f"   Agents: {executor.metadata['environment_info']['num_agents']}")
        print(f"   Grid: {executor.metadata['environment_info']['grid_x']}x{executor.metadata['environment_info']['grid_y']}")
        return True
    except Exception as e:
        print(f"❌ Executor initialization failed: {e}")
        return False


def test_single_episode_execution():
    """Test single episode execution."""
    print("\n🎬 Testing single episode execution...")
    
    try:
        executor = ModelExecutor(auto_find_latest=True)
        
        # Execute a short episode
        frames, stats = executor.execute_episode(max_steps=10, render=True)
        
        print(f"✅ Single episode execution completed!")
        print(f"   Steps: {stats['steps']}")
        print(f"   Total Reward: {stats['total_reward']:.2f}")
        print(f"   Action Distribution: {stats['action_distribution']}")
        
        return True
    except Exception as e:
        print(f"❌ Single episode execution failed: {e}")
        return False


def test_animation_creation():
    """Test animation creation."""
    print("\n🎬 Testing animation creation...")
    
    try:
        executor = ModelExecutor(auto_find_latest=True)
        
        # Execute episode and create animation
        frames, stats = executor.execute_episode(max_steps=5, render=True)
        animation_path = executor.create_animation(frames, fps=3)
        
        print(f"✅ Animation created successfully!")
        print(f"   Animation path: {animation_path}")
        print(f"   Frames: {len(frames)}")
        
        return True
    except Exception as e:
        print(f"❌ Animation creation failed: {e}")
        return False


def test_multi_episode_evaluation():
    """Test multi-episode evaluation."""
    print("\n🔬 Testing multi-episode evaluation...")
    
    try:
        executor = ModelExecutor(auto_find_latest=True)
        
        # Run evaluation
        results = executor.run_evaluation(num_episodes=2, max_steps=5, fps=3)
        
        print(f"✅ Multi-episode evaluation completed!")
        print(f"   Episodes: {results['completed_episodes']}/{results['num_episodes']}")
        print(f"   Average Reward: {results['average_reward']:.2f}")
        print(f"   Animations: {len(results['animation_paths'])}")
        
        return True
    except Exception as e:
        print(f"❌ Multi-episode evaluation failed: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid run."""
    print("\n⚠️  Testing error handling...")
    
    try:
        # Try to initialize with non-existent run
        executor = ModelExecutor(run_path="non_existent_run", auto_find_latest=False)
        print("❌ Should have failed with non-existent run")
        return False
    except Exception as e:
        print(f"✅ Error handling works correctly: {type(e).__name__}")
        return True


def test_different_input_formats():
    """Test different input formats."""
    print("\n📝 Testing different input formats...")
    
    latest_run = find_latest_run()
    if latest_run is None:
        print("❌ No runs found for testing input formats")
        return False
    
    run_number = latest_run.name.split('_')[1]
    
    # Test different input formats
    formats_to_test = [
        run_number,  # Just number
        f"run_{run_number}",  # run_XX format
        str(latest_run),  # Full path
    ]
    
    for i, run_path in enumerate(formats_to_test):
        try:
            executor = ModelExecutor(run_path=run_path, auto_find_latest=False)
            print(f"✅ Format {i+1} works: {run_path}")
        except Exception as e:
            print(f"❌ Format {i+1} failed: {run_path} - {e}")
            return False
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Robust Execution System")
    print("=" * 50)
    
    tests = [
        ("Latest Run Detection", test_latest_run_detection),
        ("Executor Initialization", test_executor_initialization),
        ("Single Episode Execution", test_single_episode_execution),
        ("Animation Creation", test_animation_creation),
        ("Multi-Episode Evaluation", test_multi_episode_evaluation),
        ("Error Handling", test_error_handling),
        ("Different Input Formats", test_different_input_formats),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Execution system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the execution system.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 