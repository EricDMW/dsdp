#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of Wireless Communication Environment

This script demonstrates how to use the wireless communication environment
with different configurations and network architectures.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wireless_comm.config import get_args
from wireless_comm.trainer import WirelessTrainer
from wireless_comm.networks import Agent, Critic
from wireless_comm.environment import WirelessEnvironment
from wireless_comm.utils import find_neighbors, image_background


def create_default_args():
    """Create default arguments for examples."""
    class Args:
        def __init__(self):
            # Environment parameters
            self.gym_id = "wireless_comm"
            self.grid_size = 5
            self.pkg_p = 0.5
            self.success_p = 0.8
            self.ddl = 2
            self.max_cycles = 12
            self.n_obs_neighbors = 1
            
            # Training parameters
            self.test_every = 20
            self.actor_lr = 5e-4
            self.critic_lr = 5e-4
            self.seed = 1
            self.cpu = False
            self.total_timesteps = 800
            self.torch_deterministic = True
            self.track = False
            self.debug = False
            self.anneal_lr = False
            
            # Algorithm specific arguments
            self.n_sample_traj = 15
            self.n_sample_Q_steps = 256
            self.target_network_frequency = 1
            self.tau = 0.005
            self.gamma = 0.7
            self.reward_scale = 1
            self.update_epochs = 1
            self.max_grad_norm = 1
            
            # CDSAC specific arguments
            self.eta_mu = 0
            self.rhs = 2
            
            # Logging and saving
            self.log_path = "training_log.csv"
            self.save_dir = "hyperparameters"
            self.experiment_name = "wireless_dsdp"
            
            # Network architecture
            self.model = "mlp"
            self.hidden_size = 256
            self.num_layers = 3
    
    return Args()


def example_basic_training():
    """Example of basic training with default parameters."""
    print("=== Basic Training Example ===")
    
    # Create default arguments
    args = create_default_args()
    args.total_timesteps = 100  # Small number for quick demo
    args.track = False  # Disable wandb for demo
    args.log_path = "example_training_log.csv"
    args.save_dir = "example_hyperparameters"
    args.experiment_name = "example_basic"
    
    # Create and run trainer
    trainer = WirelessTrainer(args)
    trainer.train()


def example_custom_network():
    """Example with custom network architecture."""
    print("=== Custom Network Example ===")
    
    # Create arguments with custom network
    args = create_default_args()
    args.total_timesteps = 50  # Small number for quick demo
    args.model = "mlp"
    args.hidden_size = 128
    args.num_layers = 2
    args.grid_size = 3  # Smaller grid for faster demo
    args.track = False
    args.log_path = "example_custom_network_log.csv"
    args.save_dir = "example_hyperparameters"
    args.experiment_name = "example_custom_network"
    
    # Create and run trainer
    trainer = WirelessTrainer(args)
    trainer.train()


def example_debug_mode():
    """Example with debug mode enabled."""
    print("=== Debug Mode Example ===")
    
    # Create arguments with debug mode
    args = create_default_args()
    args.total_timesteps = 30  # Very small number for quick demo
    args.debug = True
    args.test_every = 10
    args.grid_size = 3
    args.track = False
    args.log_path = "example_debug_log.csv"
    args.save_dir = "example_hyperparameters"
    args.experiment_name = "example_debug"
    
    # Create and run trainer
    trainer = WirelessTrainer(args)
    trainer.train()


def example_environment_only():
    """Example of just using the environment without training."""
    print("=== Environment Only Example ===")
    
    import torch
    
    # Create arguments
    args = create_default_args()
    args.grid_size = 3
    args.ddl = 2
    args.n_obs_neighbors = 1
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = WirelessEnvironment(args, device)
    env_info = env.get_environment_info()
    
    print(f"Environment created successfully!")
    print(f"Grid size: {env_info['grid_x']}x{env_info['grid_y']}")
    print(f"Number of agents: {env_info['num_agents']}")
    print(f"Number of actions: {env_info['num_actions']}")
    print(f"Number of states: {env_info['num_states']}")
    
    # Run a few steps
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else 'dict'}")
    
    for step in range(5):
        # Get random actions
        actions = torch.randint(0, env_info['num_actions'], (env_info['num_agents'],))
        
        # Take step
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        
        print(f"Step {step}: Rewards = {rewards}")
        
        if env.is_episode_done(terms, truncs):
            print("Episode ended!")
            obs, info = env.reset()
        else:
            obs = next_obs
    
    env.close()
    print("Environment test completed!")


def example_network_only():
    """Example of just using the networks without training."""
    print("=== Network Only Example ===")
    
    import torch
    
    # Create arguments
    args = create_default_args()
    args.model = "mlp"
    args.hidden_size = 64
    args.num_layers = 2
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create networks
    num_states = 9  # 3x3 grid
    num_actions = 5
    num_ddl = 2
    n_obs_neighbors = 1
    
    agent = Agent(num_states, num_actions, num_ddl, args.model, args.hidden_size, args.num_layers)
    critic = Critic(num_states, num_actions, n_obs_neighbors, num_ddl, args.model, args.hidden_size, args.num_layers)
    
    agent.to(device)
    critic.to(device)
    
    print(f"Agent created: {agent}")
    print(f"Critic created: {critic}")
    
    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, num_ddl, num_states).to(device)
    actions = torch.randint(0, num_actions, (batch_size,)).to(device)
    
    # Test agent
    agent_actions, log_probs, entropy = agent.get_action(obs)
    print(f"Agent output - Actions: {agent_actions.shape}, Log probs: {log_probs.shape}, Entropy: {entropy.shape}")
    
    # Test critic
    critic_values = critic.get_value(obs, actions)
    print(f"Critic output - Values: {critic_values.shape}")
    
    print("Network test completed!")


def example_direct_env_usage():
    """Example of using the existing WirelessCommEnv directly."""
    print("=== Direct Environment Usage Example ===")
    
    from env_lib import WirelessCommEnv
    import time
    
    # Create environment directly
    env = WirelessCommEnv(
        grid_x=4, 
        grid_y=4, 
        ddl=3, 
        render_mode="rgb_array",
        max_iter=200, 
        save_gif=True, 
        gif_path="example_run.gif",
        debug_info=True,
        n_obs_neighbors=1
    )
    
    obs, info = env.reset()
    time_start = time.time()
    
    print("Running environment for 30 steps...")
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()  # Shows animation in real time
        time.sleep(0.1)  # Reduced sleep for faster demo
        print(f"Step {step + 1}: Reward = {reward}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()  # Saves GIF if save_gif=True
    time_end = time.time()
    
    print(f"Environment run completed in {time_end - time_start:.2f} seconds")
    print("Check 'example_run.gif' for the saved animation!")


def example_parameter_tuning():
    """Example of using the ParameterTuner for hyperparameter tuning."""
    print("=== Parameter Tuning Example ===")
    
    import argparse
    from pathlib import Path
    
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
        params_dir = Path("example_parameters")
        params_dir.mkdir(exist_ok=True)
        
        # Configure ParameterTuner with validation
        tuner = ParameterTuner(
            parser=parser,
            save_path=str(params_dir),
            save_delay=10000,  # 10 seconds auto-save for demo
            auto_save_enabled=True,
            inactivity_timeout=10,  # 10 seconds inactivity timeout for demo
            validation_callbacks={
                'grid_size': lambda x: 2 <= x <= 10,
                'learning_rate': lambda x: 1e-6 <= x <= 1e-1,
                'batch_size': lambda x: 1 <= x <= 128,
                'epochs': lambda x: 1 <= x <= 1000,
                'hidden_size': lambda x: 16 <= x <= 1024,
                'dropout': lambda x: 0.0 <= x <= 0.9
            }
        )
        
        # Launch parameter tuning GUI
        print("GUI will open for parameter tuning...")
        print("You can modify parameters and they will be auto-saved.")
        print("The GUI will close automatically after 10 seconds of inactivity.")
        
        updated_parser = tuner.tune()
        args = updated_parser.parse_args()
        
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
            import json
            with open(latest_file, 'r') as f:
                saved_params = json.load(f)
            print(f"Saved parameters: {saved_params}")
        
        print("Parameter tuning example completed!")
        
    except ImportError:
        print("ParameterTuner not available. Skipping parameter tuning example.")
    except Exception as e:
        print(f"Parameter tuning failed: {e}")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Wireless Communication Examples")
    parser.add_argument("--example", type=str, default="direct_env", 
                       choices=["basic", "custom_network", "debug", "env_only", "network_only", "direct_env", "parameter_tuning"],
                       help="Which example to run")
    
    args = parser.parse_args()
    
    if args.example == "basic":
        example_basic_training()
    elif args.example == "custom_network":
        example_custom_network()
    elif args.example == "debug":
        example_debug_mode()
    elif args.example == "env_only":
        example_environment_only()
    elif args.example == "network_only":
        example_network_only()
    elif args.example == "direct_env":
        example_direct_env_usage()
    elif args.example == "parameter_tuning":
        example_parameter_tuning()
    else:
        print(f"Unknown example: {args.example}")
        return
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main() 