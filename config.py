#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration and Argument Parsing for Wireless Communication

This module provides argument parsing and parameter tuning using the toolkit's ParameterTuner.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path

try:
    from toolkit.parakit import ParameterTuner
    TOOLKIT_AVAILABLE = True
except ImportError:
    TOOLKIT_AVAILABLE = False
    print("Warning: toolkit.parakit not available. Using basic argument parsing.")


def get_args():
    """Get command line arguments with parameter tuning support."""
    parser = argparse.ArgumentParser(description="Wireless Communication Environment Training")
    
    # Environment parameters
    parser.add_argument("--gym-id", type=str, default="wireless_comm",
                       help="the id of the gym environment")
    parser.add_argument("--grid-size", type=int, default=5,
                       help="grid size for the environment")
    parser.add_argument("--pkg-p", type=float, default=0.5,
                       help="package arrival probability")
    parser.add_argument("--success-p", type=float, default=0.8,
                       help="success transmission probability")
    parser.add_argument("--ddl", type=int, default=2,
                       help="deadline horizon")
    parser.add_argument("--max-cycles", type=int, default=20,
                       help="maximum number of cycles per episode")
    parser.add_argument("--n-obs-neighbors", type=int, default=1,
                       help="number of neighbors to observe")
    
    # Training parameters
    parser.add_argument("--test-every", type=int, default=20,
                       help="test every N updates")
    parser.add_argument("--actor-lr", type=float, default=5e-4,
                       help="the learning rate of the actor network")
    parser.add_argument("--critic-lr", type=float, default=5e-4,
                       help="the learning rate of the critic network")
    parser.add_argument("--seed", type=int, default=1,
                       help="seed of the experiment")
    parser.add_argument("--cpu", type=lambda x: bool(strtobool(x)), default=False, help="if toggled, cuda will not be enabled by default")
    parser.add_argument("--total-timesteps", type=int, default=80000,
                       help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                       help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                       help="if toggled, this experiment will be tracked with Weights & Biases")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                       help="if toggled, debug mode will be enabled")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                       help="Toggle learning rate annealing for policy and value networks")
    
    # Algorithm specific arguments
    parser.add_argument("--n-sample-traj", type=int, default=15,
                       help="number of trajectories to sample")
    parser.add_argument("--n-sample-Q-steps", type=int, default=256,
                       help="number of Q-function update steps")
    parser.add_argument("--target-network-frequency", type=int, default=1,
                       help="the frequency of updates for the target networks")
    parser.add_argument("--policy-frequency", type=int, default=1, help="the frequency of policy updates")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="noise clip value for policy smoothing")
    parser.add_argument("--tau", type=float, default=0.005,
                       help="the target network update rate")
    parser.add_argument("--gamma", type=float, default=0.7,
                       help="the discount factor gamma")
    parser.add_argument("--reward-scale", type=float, default=1,
                       help="reward scale factor")
    parser.add_argument("--update-epochs", type=int, default=1,
                       help="the K epochs to update the policy")
    parser.add_argument("--max-grad-norm", type=float, default=1,
                       help="the maximum norm for the gradient clipping")
    parser.add_argument("--learning-starts", type=int, default=0,
                       help="timesteps to run collecting random policy before learning starts")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="batch size for training")
    parser.add_argument("--log-freq", type=int, default=10,
                       help="frequency of logging")
    parser.add_argument("--buffer-size", type=int, default=10000,
                       help="size of replay buffer")
    
    # CDSAC specific arguments
    parser.add_argument("--eta-mu", type=float, default=2,
                       help="constraint weight") # TODO: ablation
    parser.add_argument("--rhs", type=float, default=2,
                       help="constraint right-hand side")
    
    # Logging and saving
    parser.add_argument("--log-path", type=str, default="training_log.csv",
                       help="path to save training log")
    parser.add_argument("--save-dir", type=str, default="hyperparameters",
                       help="directory to save hyperparameters")
    parser.add_argument("--experiment-name", type=str, default="wireless_dsdp",
                       help="name of the experiment")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory to use (created by manager)")
    
    # Network architecture
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn", "rnn", "transformer"],
                       help="neural network architecture")
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="hidden layer size")
    parser.add_argument("--num-layers", type=int, default=3,
                       help="number of hidden layers")
    
    # Performance optimization arguments
    parser.add_argument("--pin-memory", type=lambda x: bool(strtobool(x)), default=True,
                       help="use pin_memory for faster GPU transfer")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="number of steps to accumulate gradients before updating")
    parser.add_argument("--mixed-precision", type=lambda x: bool(strtobool(x)), default=False,
                       help="use mixed precision training for faster computation")
    parser.add_argument("--profile-training", type=lambda x: bool(strtobool(x)), default=False,
                       help="profile training for performance analysis")
    
    # === Additional arguments required by trainer.py ===
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy regularization coefficient")
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=False, help="automatic entropy tuning")
    parser.add_argument("--grad-norm", type=float, default=0.5, help="gradient norm clipping value")
    parser.add_argument("--target-policy-noise", type=float, default=0.2, help="target policy noise for TD3")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="policy noise for TD3")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--q-hidden-size", type=int, default=256, help="Q-network hidden size")
    parser.add_argument("--p-hidden-size", type=int, default=256, help="Policy network hidden size")
    parser.add_argument("--n-critics", type=int, default=2, help="number of critics")
    parser.add_argument("--use-sde", type=lambda x: bool(strtobool(x)), default=False, help="use State Dependent Exploration")
    parser.add_argument("--sde-sample-freq", type=int, default=-1, help="SDE sample frequency")
    parser.add_argument("--use-sde-at-warmup", type=lambda x: bool(strtobool(x)), default=False, help="use SDE at warmup")
    parser.add_argument("--stats-window-size", type=int, default=100, help="Stats window size")
    parser.add_argument("--tensorboard-log", type=str, default="", help="Tensorboard log dir")
    parser.add_argument("--policy-kwargs", type=str, default="{}", help="Policy kwargs as JSON string")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False, help="Deterministic evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--_init-setup-model", type=lambda x: bool(strtobool(x)), default=True, help="Init setup model")
    parser.add_argument("--wandb-project-name", type=str, default="", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default="", help="WandB entity")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, help="Capture video")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, help="Upload model")
    parser.add_argument("--hf-repo", type=str, default="", help="HuggingFace repo")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save frequency")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Eval frequency")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--num-eval-episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--optimize-memory-usage", type=lambda x: bool(strtobool(x)), default=False, help="Optimize memory usage")
    parser.add_argument("--env-kwargs", type=str, default="{}", help="Env kwargs as JSON string")
    parser.add_argument("--eval-env-kwargs", type=str, default="{}", help="Eval env kwargs as JSON string")
    parser.add_argument("--n-eval-envs", type=int, default=1, help="Number of eval envs")
    parser.add_argument("--dummy-vec-env", type=lambda x: bool(strtobool(x)), default=False, help="Dummy VecEnv")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of envs")
    parser.add_argument("--monitor-dir", type=str, default="", help="Monitor dir")
    parser.add_argument("--filename", type=str, default="", help="Filename")
    parser.add_argument("--vec-env-class", type=str, default="", help="VecEnv class")
    parser.add_argument("--vec-env-kwargs", type=str, default="{}", help="VecEnv kwargs as JSON string")
    parser.add_argument("--monitor-kwargs", type=str, default="{}", help="Monitor kwargs as JSON string")
    parser.add_argument("--wrapper-class", type=str, default="", help="Wrapper class")
    parser.add_argument("--env-wrapper", type=str, default="", help="Env wrapper")
    parser.add_argument("--load-best-model-at-end", type=lambda x: bool(strtobool(x)), default=False, help="Load best model at end")
    parser.add_argument("--auto-load-best-model", type=lambda x: bool(strtobool(x)), default=False, help="Auto load best model")
    # === End of additional arguments ===
    
    # Parameter tuning
    parser.add_argument("--tune-parameters", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                       help="if toggled, launch parameter tuning GUI")
    parser.add_argument("--parameters-dir", type=str, default="parameters",
                       help="directory to save parameters from GUI")
    
    # Use ParameterTuner if available
    if TOOLKIT_AVAILABLE and parser.parse_known_args()[0].tune_parameters:
        try:
            # Create parameters directory
            parameters_dir = Path(parser.parse_known_args()[0].parameters_dir)
            parameters_dir.mkdir(exist_ok=True)
            
            # Configure ParameterTuner and launch GUI
            updated_parser = ParameterTuner.tune_parameters(
                parser=parser,
                save_path=str(parameters_dir),
                save_delay=1000,  # Immediate auto-save
                auto_save_enabled=True,
                inactivity_timeout=15,  # 15 seconds inactivity timeout
                validation_callbacks={
                    'grid_size': lambda x: (2 <= int(x) <= 10, int(x), "Must be between 2 and 10"),
                    'pkg_p': lambda x: (0.0 <= float(x) <= 1.0, float(x), "Must be between 0.0 and 1.0"),
                    'success_p': lambda x: (0.0 <= float(x) <= 1.0, float(x), "Must be between 0.0 and 1.0"),
                    'ddl': lambda x: (1 <= int(x) <= 5, int(x), "Must be between 1 and 5"),
                    'max_cycles': lambda x: (5 <= int(x) <= 50, int(x), "Must be between 5 and 50"),
                    'n_obs_neighbors': lambda x: (1 <= int(x) <= 4, int(x), "Must be between 1 and 4"),
                    'actor_lr': lambda x: (1e-6 <= float(x) <= 1e-2, float(x), "Must be between 1e-6 and 1e-2"),
                    'critic_lr': lambda x: (1e-6 <= float(x) <= 1e-2, float(x), "Must be between 1e-6 and 1e-2"),
                    'gamma': lambda x: (0.1 <= float(x) <= 0.99, float(x), "Must be between 0.1 and 0.99"),
                    'eta_mu': lambda x: (0.0 <= float(x) <= 10.0, float(x), "Must be between 0.0 and 10.0"),
                    'rhs': lambda x: (0.0 <= float(x) <= 10.0, float(x), "Must be between 0.0 and 10.0"),
                    'hidden_size': lambda x: (32 <= int(x) <= 1024, int(x), "Must be between 32 and 1024"),
                    'num_layers': lambda x: (1 <= int(x) <= 10, int(x), "Must be between 1 and 10")
                }
            )
            return updated_parser.parse_args()
            
        except Exception as e:
            print(f"Warning: Parameter tuning failed: {e}")
            print("Falling back to command line arguments...")
            return parser.parse_args()
    else:
        return parser.parse_args()


def save_hyperparameters(args, save_dir="hyperparameters"):
    """Save hyperparameters to a JSON file."""
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hyperparameters_{timestamp}.json"
    filepath = save_path / filename
    
    # Convert args to dictionary
    params = vars(args)
    
    # Add metadata
    params['timestamp'] = timestamp
    params['experiment_name'] = getattr(args, 'experiment_name', 'wireless_dsdp')
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4, default=str)
    
    print(f"Hyperparameters saved to: {filepath}")
    return filepath


def load_hyperparameters(filepath, parser=None):
    """Load hyperparameters from a JSON file, with type conversion if parser is provided."""
    with open(filepath, 'r') as f:
        params = json.load(f)
    
    args = argparse.Namespace()
    if parser is not None:
        for action in parser._actions:
            if action.dest in params and action.dest != 'help':
                value = params[action.dest]
                # Convert to correct type
                if action.type:
                    try:
                        value = action.type(value)
                    except Exception:
                        pass  # fallback to original if conversion fails
                elif action.choices:
                    if value not in action.choices:
                        value = action.default
                setattr(args, action.dest, value)
    else:
        for key, value in params.items():
            if key not in ['timestamp', 'experiment_name']:
                setattr(args, key, value)
    return args


def strtobool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def print_hyperparameters(args):
    """Print all hyperparameters in a formatted way."""
    print("\n" + "="*60)
    print("HYPERPARAMETERS")
    print("="*60)
    
    # Environment parameters
    print("\n🌍 ENVIRONMENT PARAMETERS:")
    print(f"  Grid Size: {args.grid_size}")
    print(f"  Package Arrival Probability: {args.pkg_p}")
    print(f"  Success Transmission Probability: {args.success_p}")
    print(f"  Deadline Horizon: {args.ddl}")
    print(f"  Max Cycles: {args.max_cycles}")
    print(f"  Neighbors to Observe: {args.n_obs_neighbors}")
    
    # Training parameters
    print("\n🎯 TRAINING PARAMETERS:")
    print(f"  Total Timesteps: {args.total_timesteps}")
    print(f"  Actor Learning Rate: {args.actor_lr}")
    print(f"  Critic Learning Rate: {args.critic_lr}")
    print(f"  Gamma (Discount Factor): {args.gamma}")
    print(f"  Tau (Target Update Rate): {args.tau}")
    print(f"  Max Gradient Norm: {args.max_grad_norm}")
    
    # Algorithm parameters
    print("\n🧠 ALGORITHM PARAMETERS:")
    print(f"  Number of Sample Trajectories: {args.n_sample_traj}")
    print(f"  Number of Q-Steps: {args.n_sample_Q_steps}")
    print(f"  Update Epochs: {args.update_epochs}")
    print(f"  Eta Mu (Constraint Weight): {args.eta_mu}")
    print(f"  RHS (Constraint Right-Hand Side): {args.rhs}")
    
    # Network parameters
    print("\n🕸️ NETWORK PARAMETERS:")
    print(f"  Model Architecture: {args.model.upper()}")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  Number of Layers: {args.num_layers}")
    
    # Logging parameters
    print("\n📊 LOGGING PARAMETERS:")
    print(f"  Log Path: {args.log_path}")
    print(f"  Save Directory: {args.save_dir}")
    print(f"  Experiment Name: {args.experiment_name}")
    print(f"  Track with WandB: {args.track}")
    print(f"  Debug Mode: {args.debug}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the configuration
    args = get_args()
    print_hyperparameters(args)
    
    # Save hyperparameters
    save_path = save_hyperparameters(args)
    print(f"Hyperparameters saved to: {save_path}") 