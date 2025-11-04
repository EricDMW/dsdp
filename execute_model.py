#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Execution and Animation Generation

This module provides functionality to execute trained models and generate GIF animations.
It can automatically find the most recent run or load from a specific run directory.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import argparse
import json
import sys
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Decentralized-Safe-MARL-with-General-Utilities'))

# Import local modules
try:
    from .networks import Agent
    from .environment import WirelessEnvironment
    from .utils import name_with_datetime
except ImportError:
    # Fallback for direct execution
    from networks import Agent
    from environment import WirelessEnvironment
    from utils import name_with_datetime


class ModelExecutor:
    """Execute trained models and generate animations."""
    
    def __init__(self, run_path: Optional[str] = None, device: str = "cpu", auto_find_latest: bool = True):
        """
        Initialize the model executor.
        
        Args:
            run_path: Path to the run directory (e.g., "dsdp/wireless_comm/runs/run_1")
                     If None and auto_find_latest=True, will find the most recent run
            device: Device to run on (cpu/cuda)
            auto_find_latest: Whether to automatically find the latest run if run_path is None
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Determine run path
        if run_path is None and auto_find_latest:
            self.run_path = self._find_latest_run()
            print(f"🔍 Auto-detected latest run: {self.run_path.name}")
        else:
            self.run_path = self._resolve_run_path(run_path)
        
        # Setup logging
        self._setup_logging()
        
        # Load model metadata and configuration
        self.metadata = self._load_metadata()
        self.hyperparameters = self._load_hyperparameters()
        
        # Load trained models
        self.agents = self._load_agents()
        
        # Setup environment
        self.env_wrapper = self._setup_environment()
        
        # Setup execution directories
        self._setup_execution_dirs()
        
        print(f"✅ Model executor initialized for: Run {self.metadata['run_number']}")
        print(f"   Device: {self.device}")
        print(f"   Agents: {self.metadata['environment_info']['num_agents']}")
        print(f"   Grid: {self.metadata['environment_info']['grid_x']}x{self.metadata['environment_info']['grid_y']}")
        print(f"   Model ID: {self.metadata['model_id']}")
    
    def _find_latest_run(self) -> Path:
        """Find the most recent run directory."""
        runs_dir = Path(__file__).parent / "runs"
        
        if not runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        
        # Find all run directories
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        
        if not run_dirs:
            raise FileNotFoundError("No run directories found. Please run training first.")
        
        # Find the most recent run by creation time
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        
        # Verify the run has required files
        if not self._validate_run_directory(latest_run):
            raise ValueError(f"Run directory {latest_run.name} is incomplete. Missing required files.")
        
        return latest_run
    
    def _resolve_run_path(self, run_path: Optional[str]) -> Path:
        """Resolve run path from various input formats."""
        if run_path is None:
            raise ValueError("run_path cannot be None when auto_find_latest=False")
        
        # Handle different input formats
        if run_path.isdigit():
            # Just a number
            run_dir = Path(__file__).parent / "runs" / f"run_{run_path}"
        elif run_path.startswith('run_'):
            # run_XX format
            run_dir = Path(__file__).parent / "runs" / run_path
        elif run_path.startswith('runs/run_'):
            # runs/run_XX format
            run_dir = Path(__file__).parent / run_path
        else:
            # Full path
            run_dir = Path(run_path)
        
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        # Validate the run directory
        if not self._validate_run_directory(run_dir):
            raise ValueError(f"Run directory {run_dir.name} is incomplete. Missing required files.")
        
        return run_dir
    
    def _validate_run_directory(self, run_dir: Path) -> bool:
        """Validate that a run directory has all required files."""
        required_files = [
            "model/training_metadata.json",
            "hyperparameters"
        ]
        
        for file_path in required_files:
            if not (run_dir / file_path).exists():
                return False
        
        # Check for at least one policy model
        model_dir = run_dir / "model"
        policy_files = list(model_dir.glob("agent_*_policy.pt"))
        if not policy_files:
            return False
        
        return True
    
    def _setup_logging(self):
        """Setup logging for execution."""
        self.logger = logging.getLogger(f"ModelExecutor_{self.run_path.name}")
        self.logger.setLevel(logging.INFO)
        
        # Create execution directory for logs
        execution_dir = self.run_path / "execution"
        execution_dir.mkdir(exist_ok=True)
        
        # Add file handler
        log_file = execution_dir / "execution.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _load_metadata(self) -> Dict:
        """Load training metadata."""
        metadata_path = self.run_path / "model" / "training_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Loaded metadata from: {metadata_path}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            raise
    
    def _load_hyperparameters(self) -> Dict:
        """Load hyperparameters from the run directory."""
        hyperparams_dir = self.run_path / "hyperparameters"
        
        if not hyperparams_dir.exists():
            self.logger.warning("Hyperparameters directory not found, using metadata config")
            return self.metadata.get('training_config', {})
        
        # Find the most recent hyperparameters file
        hyperparam_files = list(hyperparams_dir.glob("hyperparameters_*.json"))
        
        if not hyperparam_files:
            self.logger.warning("No hyperparameters files found, using metadata config")
            return self.metadata.get('training_config', {})
        
        # Get the most recent file
        latest_file = max(hyperparam_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                hyperparams = json.load(f)
            
            self.logger.info(f"Loaded hyperparameters from: {latest_file}")
            return hyperparams
        except Exception as e:
            self.logger.error(f"Failed to load hyperparameters: {e}")
            return self.metadata.get('training_config', {})
    
    def _load_agents(self) -> List[Agent]:
        """Load trained agent models."""
        num_agents = self.metadata['environment_info']['num_agents']
        agents = []
        
        model_dir = self.run_path / "model"
        
        for i in range(num_agents):
            model_path = model_dir / f"agent_{i+1}_policy.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Agent model not found: {model_path}")
            
            try:
                # Load model checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                model_config = checkpoint['model_config']
                
                # Get actual input size from the saved model
                state_dict = checkpoint['model_state_dict']
                if 'policy_network.feature_layers.0.weight' in state_dict:
                    actual_input_size = state_dict['policy_network.feature_layers.0.weight'].shape[1]
                    self.logger.info(f"Model config says num_states={model_config['num_states']}, but actual input size is {actual_input_size}")
                    # Use the actual input size from the saved model
                    model_config['num_states'] = actual_input_size
                
                # Create agent with correct parameters
                agent = Agent(
                    num_states=model_config['num_states'],
                    num_actions=model_config['num_actions'],
                    num_ddl=model_config['num_ddl'],
                    model_type=model_config['model_type'],
                    hidden_size=model_config['hidden_size'],
                    num_layers=model_config['num_layers']
                )
                
                # Load state dict
                agent.load_state_dict(checkpoint['model_state_dict'])
                agent.to(self.device)
                agent.eval()
                
                agents.append(agent)
                self.logger.info(f"Loaded agent_{i+1}_policy.pt (input_size={model_config['num_states']})")
                
            except Exception as e:
                error_msg = f"Failed to load agent_{i+1}: {e}"
                self.logger.error(error_msg)
                
                # Save detailed error to execution directory
                error_dir = self.run_path / "execution"
                error_dir.mkdir(exist_ok=True)
                error_log = error_dir / "model_loading_errors.log"
                
                with open(error_log, 'a') as f:
                    f.write(f"{name_with_datetime()}: {error_msg}\n")
                
                raise RuntimeError(error_msg)
        
        return agents
    
    def _setup_environment(self) -> WirelessEnvironment:
        """Setup environment with training configuration."""
        # Create argument namespace from metadata and hyperparameters
        class Args:
            def __init__(self, metadata, hyperparams):
                # Use hyperparameters if available, otherwise fall back to metadata
                config = hyperparams if hyperparams else metadata['training_config']
                
                self.grid_size = int(config.get('grid_size', 5))
                self.pkg_p = float(config.get('pkg_p', 0.5))
                self.success_p = float(config.get('success_p', 0.8))
                self.ddl = int(config.get('ddl', 2))
                self.max_cycles = int(config.get('max_cycles', 30))
                self.n_obs_neighbors = int(config.get('n_obs_neighbors', 1))
        
        args = Args(self.metadata, self.hyperparameters)
        return WirelessEnvironment(args, self.device)
    
    def _setup_execution_dirs(self):
        """Setup directories for execution output."""
        self.execution_dir = self.run_path / "execution"
        self.execution_dir.mkdir(exist_ok=True)
    
    def execute_episode(self, max_steps: Optional[int] = None, render: bool = True, 
                       save_frames: bool = True) -> Tuple[List[np.ndarray], Dict]:
        """
        Execute a single episode with the trained model.
        
        Args:
            max_steps: Maximum steps to execute (None for environment default)
            render: Whether to render frames
            save_frames: Whether to save individual frames
            
        Returns:
            Tuple of (frames, episode_stats)
        """
        if max_steps is None:
            max_steps = self.metadata['environment_info']['max_cycles']
        
        # Reset environment
        obs, info = self.env_wrapper.reset()
        frames = []
        episode_stats = {
            'total_reward': 0,
            'steps': 0,
            'successful_transmissions': 0,
            'failed_transmissions': 0,
            'constraint_violations': 0,
            'action_distribution': {}
        }
        
        self.logger.info(f"Starting episode execution (max {max_steps} steps)")
        
        for step in range(max_steps):
            try:
                # Get observations
                obs_batch = self.env_wrapper.get_batchified_obs(obs)
                
                # Get actions from agents
                actions = self.env_wrapper.get_agent_actions(obs_batch, self.agents)
                
                # Take step in environment
                next_obs, rewards, terms, truncs, infos = self.env_wrapper.step(actions)
                
                # Update statistics
                rewards_batch = self.env_wrapper.get_batchified_rewards(rewards)
                episode_stats['total_reward'] += rewards_batch.sum().item()
                episode_stats['steps'] = step + 1
                
                # Count transmissions and update action distribution
                action_counts = torch.bincount(actions, minlength=5)
                for action_id, count in enumerate(action_counts):
                    if action_id not in episode_stats['action_distribution']:
                        episode_stats['action_distribution'][action_id] = 0
                    episode_stats['action_distribution'][action_id] += count.item()
                
                # Count successful/failed transmissions
                # Handle different reward formats
                if rewards_batch.dim() == 0:
                    # Scalar tensor - no individual agent rewards
                    self.logger.debug(f"Scalar reward at step {step}: {rewards_batch.item()}")
                elif rewards_batch.dim() == 1 and len(rewards_batch) > 0:
                    # 1D tensor with individual agent rewards
                    for i, action in enumerate(actions):
                        if action > 0:  # Non-idle action
                            if i < len(rewards_batch) and rewards_batch[i] > 0:
                                episode_stats['successful_transmissions'] += 1
                            else:
                                episode_stats['failed_transmissions'] += 1
                else:
                    # Empty or unexpected format
                    self.logger.warning(f"Unexpected rewards format at step {step}: shape={rewards_batch.shape}")
                
                # Render frame if requested
                if render:
                    frame = self._render_frame(obs_batch, actions, rewards_batch, step)
                    frames.append(frame)
                    
                    if save_frames:
                        frame_path = self.execution_dir / f"frame_{step:04d}.png"
                        cv2.imwrite(str(frame_path), frame)
                
                # Update observations
                obs = next_obs
                
                # Check if episode is done
                if self.env_wrapper.is_episode_done(terms, truncs):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error during step {step}: {e}")
                raise
        
        self.logger.info(f"Episode completed in {episode_stats['steps']} steps")
        self.logger.info(f"Total Reward: {episode_stats['total_reward']:.2f}")
        self.logger.info(f"Successful Transmissions: {episode_stats['successful_transmissions']}")
        self.logger.info(f"Failed Transmissions: {episode_stats['failed_transmissions']}")
        
        return frames, episode_stats
    
    def _render_frame(self, obs: torch.Tensor, actions: torch.Tensor, 
                     rewards: torch.Tensor, step: int) -> np.ndarray:
        """
        Render a single frame of the environment.
        
        Args:
            obs: Current observations
            actions: Actions taken by agents
            rewards: Rewards received
            step: Current step number
            
        Returns:
            Rendered frame as numpy array
        """
        try:
            # Get environment render
            env_frame = self.env_wrapper.render()
            
            # Add text overlay
            frame = env_frame.copy()
            
            # Add step counter
            cv2.putText(frame, f"Step: {step}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add total reward
            total_reward = rewards.sum().item()
            cv2.putText(frame, f"Reward: {total_reward:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add action distribution
            action_counts = torch.bincount(actions, minlength=5)
            action_text = f"Actions: {action_counts.tolist()}"
            cv2.putText(frame, action_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"Error rendering frame: {e}")
            # Return a simple black frame with text
            frame = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(frame, f"Step: {step}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
    
    def create_animation(self, frames: List[np.ndarray], fps: int = 3, 
                        save_path: Optional[str] = None) -> str:
        """
        Create GIF animation from frames.
        
        Args:
            frames: List of frames as numpy arrays
            fps: Frames per second
            save_path: Path to save animation (auto-generated if None)
            
        Returns:
            Path to saved animation
        """
        if not frames:
            raise ValueError("No frames provided for animation")
        
        if save_path is None:
            timestamp = name_with_datetime()
            model_name = self.metadata['model_id']
            save_path = self.execution_dir / f"{model_name}_execution_{timestamp}.gif"
        
        self.logger.info(f"Creating animation with {len(frames)} frames at {fps} FPS")
        
        try:
            # Convert frames to PIL Images
            pil_frames = []
            for frame in frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                pil_frames.append(pil_frame)
            
            # Save as GIF
            pil_frames[0].save(
                save_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=1000//fps,  # Duration in milliseconds
                loop=0
            )
            
            self.logger.info(f"Animation saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating animation: {e}")
            raise
    
    def run_evaluation(self, num_episodes: int = 5, max_steps: Optional[int] = None, 
                      fps: int = 3) -> Dict:
        """
        Run evaluation on multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            fps: Frames per second for animations
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info(f"Running evaluation on {num_episodes} episodes")
        
        all_stats = []
        all_animations = []
        
        for episode in range(num_episodes):
            self.logger.info(f"Episode {episode + 1}/{num_episodes}")
            
            try:
                # Execute episode
                frames, stats = self.execute_episode(max_steps=max_steps, render=True)
                
                # Create animation
                animation_path = self.create_animation(frames, fps=fps)
                
                all_stats.append(stats)
                all_animations.append(animation_path)
                
            except Exception as e:
                self.logger.error(f"Error in episode {episode + 1}: {e}")
                continue
        
        # Calculate aggregate statistics
        if all_stats:
            avg_reward = np.mean([s['total_reward'] for s in all_stats])
            avg_steps = np.mean([s['steps'] for s in all_stats])
            avg_success = np.mean([s['successful_transmissions'] for s in all_stats])
            avg_failed = np.mean([s['failed_transmissions'] for s in all_stats])
        else:
            avg_reward = avg_steps = avg_success = avg_failed = 0
        
        results = {
            'model_id': self.metadata['model_id'],
            'run_number': self.metadata['run_number'],
            'num_episodes': num_episodes,
            'completed_episodes': len(all_stats),
            'average_reward': avg_reward,
            'average_steps': avg_steps,
            'average_successful_transmissions': avg_success,
            'average_failed_transmissions': avg_failed,
            'animation_paths': all_animations,
            'episode_stats': all_stats
        }
        
        # Save evaluation results
        results_path = self.execution_dir / f"{self.metadata['model_id']}_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation completed. Results saved to: {results_path}")
        
        return results


def find_latest_run() -> Optional[Path]:
    """Find the most recent run directory."""
    runs_dir = Path(__file__).parent / "runs"
    
    if not runs_dir.exists():
        return None
    
    # Find all run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        return None
    
    # Find the most recent run by creation time
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    return latest_run


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Execute trained DSDP models")
    parser.add_argument("--run_path", type=str, default=None,
                       help="Path to the run directory or run_id (e.g., 'run_1', '23', or full path). If not specified, will use the most recent run.")
    parser.add_argument("--fps", type=int, default=3,
                       help="Frames per second for animations")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (cpu/cuda)")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Maximum steps per episode")
    parser.add_argument("--num_episodes", type=int, default=1,
                       help="Number of episodes to run")
    parser.add_argument("--no_animation", action="store_true",
                       help="Skip animation creation")
    args = parser.parse_args()

    try:
        # Initialize executor
        if args.run_path is None:
            print("🔍 No run specified, finding latest run...")
            latest_run = find_latest_run()
            if latest_run is None:
                print("❌ No run directories found. Please run training first.")
                return
            print(f"📁 Using latest run: {latest_run.name}")
            executor = ModelExecutor(run_path=str(latest_run), device=args.device)
        else:
            executor = ModelExecutor(run_path=args.run_path, device=args.device)
        
        # Run execution
        if args.num_episodes == 1:
            # Single episode
            frames, stats = executor.execute_episode(max_steps=args.max_steps)
            
            if not args.no_animation:
                gif_path = executor.execution_dir / f"{executor.run_path.name}_{args.max_steps}steps.gif"
                executor.create_animation(frames, fps=args.fps, save_path=gif_path)
                print(f"🎬 Animation saved to: {gif_path}")
            
            # Save metrics
            metrics = {
                'total_reward': stats.get('total_reward', 0),
                'steps': stats.get('steps', 0),
                'successful_transmissions': stats.get('successful_transmissions', 0),
                'failed_transmissions': stats.get('failed_transmissions', 0),
                'average_reward': stats.get('total_reward', 0) / max(1, stats.get('steps', 1)),
            }
            
            import csv
            metrics_path = executor.execution_dir / "metrics.csv"
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["total_reward", "steps", "successful_transmissions", "failed_transmissions", "average_reward"])
                writer.writerow([metrics['total_reward'], metrics['steps'], 
                               metrics['successful_transmissions'], metrics['failed_transmissions'], 
                               metrics['average_reward']])
            print(f"📊 Metrics saved to: {metrics_path}")
            
        else:
            # Multiple episodes
            results = executor.run_evaluation(num_episodes=args.num_episodes, 
                                           max_steps=args.max_steps, 
                                           fps=args.fps)
            print(f"📊 Evaluation completed: {results['completed_episodes']}/{results['num_episodes']} episodes")
            print(f"   Average Reward: {results['average_reward']:.2f}")
            print(f"   Average Steps: {results['average_steps']:.1f}")
            
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 