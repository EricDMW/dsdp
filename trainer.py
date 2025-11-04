#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer for Wireless Communication Environment

This module provides the main training loop for the wireless communication environment.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import copy
import csv
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from PIL import Image

from .config import save_hyperparameters
from .networks import Agent, Critic
from .environment import WirelessEnvironment
from .utils import find_neighbors, image_background, create_visualization_frame, gather_actions_2d, gather_next_actions_2d, gather_critic_score, get_all_scores, name_with_datetime


class WirelessTrainer:
    """Main trainer for wireless communication environment."""
    
    def __init__(self, args):
        """
        Initialize the trainer.
        
        Args:
            args: Configuration arguments
        """
        self.args = args
        self.setup_device()
        self.setup_seeding()
        self.setup_logging()
        self.setup_environment()
        self.setup_networks()
        self.setup_optimizers()
        self.setup_storage()
        
        # Always set save_dir and parameters_dir to run subdirectories
        self.args.save_dir = str(self.hyperparameters_dir)
        self.args.parameters_dir = str(self.parameters_dir)
        
        # Save hyperparameters and environment parameters at initialization
        save_hyperparameters(self.args, self.args.save_dir)
        self.save_environment_parameters()
        self.episode_log_data = []  # For per-episode logging
    
    def setup_device(self):
        """Setup device (CPU/GPU)."""
        if self.args.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_seeding(self):
        """Setup random seeding."""
        if self.args.seed != -1:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.backends.cudnn.deterministic = self.args.torch_deterministic
    
    def setup_logging(self):
        """Setup logging and saving directories."""
        module_dir = Path(__file__).parent
        # Use provided run_dir if available
        if hasattr(self.args, 'run_dir') and self.args.run_dir:
            self.run_dir = Path(self.args.run_dir)
            self.runs_dir = self.run_dir.parent
            # Extract run_number from run_dir name
            try:
                self.run_number = int(self.run_dir.name.split('_')[1])
            except Exception:
                self.run_number = None
        else:
            # Create run-based directory structure
            self.runs_dir = module_dir / "runs"
            self.runs_dir.mkdir(exist_ok=True)
            # Find next run number
            self.run_number = self._get_next_run_number()
            self.run_dir = self.runs_dir / f"run_{self.run_number}"
            self.run_dir.mkdir(exist_ok=True)
        # Create subdirectories
        self.parameters_dir = self.run_dir / "parameters"
        self.hyperparameters_dir = self.run_dir / "hyperparameters"
        self.model_dir = self.run_dir / "model"
        self.training_progress_dir = self.run_dir / "training_progress"
        self.execution_dir = self.run_dir / "execution"
        self.info_dir = self.run_dir / "info"
        for subdir in [self.parameters_dir, self.hyperparameters_dir, self.model_dir, 
                      self.training_progress_dir, self.execution_dir, self.info_dir]:
            subdir.mkdir(exist_ok=True)
        # Setup CSV logging
        import csv
        self.log_data = []
        self.log_file = None
        self.log_writer = None
        log_path = getattr(self.args, "log_path", None)
        if log_path:
            self.log_file = open(str(self.training_progress_dir / log_path), mode='w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                "update", "avg_episodic_return", "violation_rate",
                "actor_grad_norm", "critic1_grad_norm", "critic2_grad_norm"
            ])
        print(f"📁 Run directory: {self.run_dir}")
        print(f"   Parameters: {self.parameters_dir}")
        print(f"   Hyperparameters: {self.hyperparameters_dir}")
        print(f"   Model: {self.model_dir}")
        print(f"   Training Progress: {self.training_progress_dir}")
        print(f"   Execution: {self.execution_dir}")
        print(f"   Info: {self.info_dir}")
    
    def setup_wandb(self):
        """Setup Weights & Biases tracking."""
        run_name = self._generate_run_name()
        wandb.init(
            entity="ANONYMOUS",
            project="ANONYMOUS",
            name=run_name,
            sync_tensorboard=True,
            monitor_gym=True,
            config=self._get_wandb_config(),
            save_code=True
        )
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
    
    def setup_environment(self):
        """Setup the wireless communication environment."""
        self.env_wrapper = WirelessEnvironment(self.args, self.device)
        env_info = self.env_wrapper.get_environment_info()
        
        self.num_agents = env_info['num_agents']
        self.num_actions = env_info['num_actions']
        self.num_states = env_info['num_states']
        self.grid_x = env_info['grid_x']
        self.grid_y = env_info['grid_y']
        self.num_ddl = env_info['num_ddl']
        self.n_obs_neighbors = env_info['n_obs_neighbors']
        self.max_cycles = env_info['max_cycles']
        
        # Setup neighbors
        self.neighbor_dict, self.n_neighbor_dict = find_neighbors(
            self.grid_x, self.grid_y, self.n_obs_neighbors
        )
        
        # Setup visualization
        self.agent_shape = 40
        self.agent_gap = 80
        self.visual_background = image_background(
            self.grid_x, self.grid_y, self.agent_shape, self.agent_gap
        )
        
        # Setup environment for parallel execution
        self.env = self.env_wrapper.env
    
    def setup_networks(self):
        """Setup neural networks."""
        # Get the actual observation size from environment
        test_obs, _ = self.env_wrapper.reset()
        test_obs_batch = self.env_wrapper.get_batchified_obs(test_obs)
        actual_obs_size = test_obs_batch.shape[1]  # Get actual observation size
        
        # Create networks with actual observation size
        self.actors = [
            Agent(actual_obs_size, self.num_actions, self.num_ddl, 
                  self.args.model, self.args.hidden_size, self.args.num_layers)
            for _ in range(self.num_agents)
        ]
        
        self.critics1 = [
            Critic(actual_obs_size, self.num_actions, self.n_obs_neighbors, self.num_ddl,
                   self.args.model, self.args.hidden_size, self.args.num_layers)
            for _ in range(self.num_agents)
        ]
        
        self.critics1_target = deepcopy(self.critics1)
        self.critics1_copy = [
            Critic(actual_obs_size, self.num_actions, self.n_obs_neighbors, self.num_ddl,
                   self.args.model, self.args.hidden_size, self.args.num_layers)
            for _ in range(self.num_agents)
        ]
        self.critics1_copy_target = deepcopy(self.critics1_copy)
        
        self.critics2 = [
            Critic(actual_obs_size, self.num_actions, self.n_obs_neighbors, self.num_ddl,
                   self.args.model, self.args.hidden_size, self.args.num_layers)
            for _ in range(self.num_agents)
        ]
        self.critics2_target = deepcopy(self.critics2)
        self.critics2_copy = [
            Critic(actual_obs_size, self.num_actions, self.n_obs_neighbors, self.num_ddl,
                   self.args.model, self.args.hidden_size, self.args.num_layers)
            for _ in range(self.num_agents)
        ]
        self.critics2_copy_target = deepcopy(self.critics2_copy)
        
        # Move to device
        for network in (self.actors + self.critics1 + self.critics1_copy + 
                       self.critics2 + self.critics2_copy):
            network.to(self.device)
        
        for network in (self.critics1_target + self.critics1_copy_target + 
                       self.critics2_target + self.critics2_copy_target):
            network.to(self.device)
            for p in network.parameters():
                p.requires_grad = False
    
    def setup_optimizers(self):
        """Setup optimizers."""
        self.actor_optimizer = [optim.Adam(x.parameters(), lr=self.args.actor_lr) for x in self.actors]
        self.critic1_optimizer = [optim.Adam(x.parameters(), lr=self.args.critic_lr) for x in self.critics1]
        self.critic1_copy_optimizer = [optim.Adam(x.parameters(), lr=self.args.critic_lr) for x in self.critics1_copy]
        self.critic2_optimizer = [optim.Adam(x.parameters(), lr=self.args.critic_lr) for x in self.critics2]
        self.critic2_copy_optimizer = [optim.Adam(x.parameters(), lr=self.args.critic_lr) for x in self.critics2_copy]
    
    def setup_storage(self):
        """Setup storage for training data."""
        self.n_sample_Q_steps = self.args.n_sample_Q_steps
        
        # Get the actual observation size from environment by testing
        test_obs, _ = self.env_wrapper.reset()
        test_obs_batch = self.env_wrapper.get_batchified_obs(test_obs)
        obs_size = test_obs_batch.shape[1]  # Get actual observation size
        
        # Q-function storage - use pin_memory for faster GPU transfer if available
        try:
            pin_memory = self.device.type == 'cuda'
            self.rb_q_obs = torch.zeros(self.n_sample_Q_steps, self.num_agents, obs_size, 
                                       device=self.device, pin_memory=pin_memory)
            self.rb_q_obs_next = torch.zeros(self.n_sample_Q_steps, self.num_agents, obs_size, 
                                            device=self.device, pin_memory=pin_memory)
            self.rb_q_actions = torch.zeros(self.n_sample_Q_steps, self.num_agents, 
                                           dtype=torch.int64, device=self.device, pin_memory=pin_memory)
            self.rb_q_rewards1 = torch.zeros(self.n_sample_Q_steps, self.num_agents, 
                                            device=self.device, pin_memory=pin_memory)
            self.rb_q_rewards2 = torch.zeros(self.n_sample_Q_steps, self.num_agents, 
                                            device=self.device, pin_memory=pin_memory)
            self.rb_q_terms = torch.zeros(self.n_sample_Q_steps, self.num_agents, 
                                         device=self.device, pin_memory=pin_memory)
        except RuntimeError:
            # Fallback to regular tensor creation if pin_memory fails
            self.rb_q_obs = torch.zeros(self.n_sample_Q_steps, self.num_agents, obs_size, 
                                       device=self.device)
            self.rb_q_obs_next = torch.zeros(self.n_sample_Q_steps, self.num_agents, obs_size, 
                                            device=self.device)
            self.rb_q_actions = torch.zeros(self.n_sample_Q_steps, self.num_agents, 
                                           dtype=torch.int64, device=self.device)
            self.rb_q_rewards1 = torch.zeros(self.n_sample_Q_steps, self.num_agents, 
                                            device=self.device)
            self.rb_q_rewards2 = torch.zeros(self.n_sample_Q_steps, self.num_agents, 
                                            device=self.device)
            self.rb_q_terms = torch.zeros(self.n_sample_Q_steps, self.num_agents, 
                                         device=self.device)
    
    def _generate_run_name(self):
        """Generate run name for tracking."""
        return f"{self.args.gym_id}_{self.num_ddl}x{self.grid_x}x{self.grid_y}_P{self.args.pkg_p}xQ{self.args.success_p}_{self.num_agents}AG_N{self.n_obs_neighbors}_Eta{self.args.eta_mu}_rhs{self.args.rhs}_seed{self.args.seed}_gm{self.args.gamma}_{name_with_datetime()}"
    
    def _generate_model_id(self):
        """Generate unique model ID with timestamp and run name."""
        timestamp = name_with_datetime()
        run_name = self._generate_run_name()
        return f"{run_name}_{timestamp}"
    
    def _get_next_run_number(self) -> int:
        """Get the next available run number."""
        existing_runs = [d for d in self.runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not existing_runs:
            return 1
        
        run_numbers = []
        for run_dir in existing_runs:
            try:
                run_num = int(run_dir.name.split("_")[1])
                run_numbers.append(run_num)
            except (ValueError, IndexError):
                continue
        
        return max(run_numbers) + 1 if run_numbers else 1
    
    def _get_wandb_config(self):
        """Get configuration for wandb."""
        return {
            'model': __file__[:__file__.find('.py')],
            'actor_lr': self.args.actor_lr,
            'critic_lr': self.args.critic_lr,
            'anneal_lr': self.args.anneal_lr,
            'gamma': self.args.gamma,
            'eta_mu': self.args.eta_mu,
            'max_cycles': self.max_cycles,
            'reward_scale': self.args.reward_scale,
            'entropy_constraint': self.args.rhs,
            'n_sample_traj': self.args.n_sample_traj,
            'n_sample_Q_steps': self.n_sample_Q_steps,
            'n_obs_neighbors': self.n_obs_neighbors,
            'update_epochs': self.args.update_epochs,
            'max_grad_norm': self.args.max_grad_norm,
        }
    
    def train(self):
        """Main training loop."""
        print(f"\n🚀 Starting training for {self.args.total_timesteps} timesteps...")
        
        # Record training start time
        import time
        training_start_time = time.time()
        update_times = []
        
        # No need to save hyperparameters/environment parameters again here
        # self.save_hyperparameters()
        # self.save_environment_parameters()
        
        # Training loop
        for update in trange(1, self.args.total_timesteps // self.args.batch_size + 1, desc='Training Progress', leave=True):
            update_start_time = time.time()
            # Learning rate annealing
            if self.args.anneal_lr:
                self._anneal_learning_rate(update)
            
            # Collect trajectories
            trajectories = self._collect_trajectories()
            
            # Calculate occupancy measure
            avg_occupancy_measure = self._calculate_occupancy_measure(trajectories)
            
            # Collect Q-function data
            self._collect_q_data(avg_occupancy_measure)
            
            # Update critics
            self._update_critics()
            
            # Update Lagrangian multiplier
            dual_mu, total_violation = self._update_lagrangian(avg_occupancy_measure)
            
            # Update actors
            self._update_actors(trajectories, dual_mu)
            
            # Log results
            self._log_results(update, trajectories, total_violation)
            
            # Debug visualization
            if self.args.debug and (update + 1) % self.args.test_every == 0:
                self._debug_visualization(update)
            
            # Log data (existing logging code)
            if update % self.args.log_freq == 0:
                # ... existing logging code ...
                pass
            
            # Track update time for performance monitoring
            update_time = time.time() - update_start_time
            update_times.append(update_time)
            
            # Print performance stats every 100 updates
            if update % 100 == 0 and update > 0:
                avg_update_time = sum(update_times[-100:]) / min(100, len(update_times))
                updates_per_sec = 1.0 / avg_update_time
                print(f"📊 Performance: {updates_per_sec:.2f} updates/sec (avg: {avg_update_time:.3f}s)")
        
        # Record training duration
        training_end_time = time.time()
        self.training_duration = training_end_time - training_start_time
        
        # Save all results
        print(f"\n💾 Saving training results...")
        self.save_training_progress()
        # Save per-episode log
        if self.episode_log_data:
            import pandas as pd
            episode_log_path = self.training_progress_dir / 'episode_log.csv'
            pd.DataFrame(self.episode_log_data).to_csv(episode_log_path, index=False)
            print(f"💾 Per-episode log saved to: {episode_log_path}")
        self.save_run_info()
        
        # Cleanup
        self.cleanup()
        
        # Save trained models
        run_number = self.save_models()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Training duration: {self.training_duration:.2f} seconds")
        print(f"📁 Run directory: {self.run_dir}")
        
        return run_number
    
    def save_models(self):
        """Save all trained models."""
        print(f"\n💾 Saving trained models to: {self.model_dir}")
        
        # Save actor networks
        for i, actor in enumerate(self.actors):
            model_path = self.model_dir / f"agent_{i+1}_policy.pt"
            torch.save({
                'model_state_dict': actor.state_dict(),
                'model_config': {
                    'num_states': self.num_states,
                    'num_actions': self.num_actions,
                    'num_ddl': self.num_ddl,
                    'model_type': self.args.model,
                    'hidden_size': self.args.hidden_size,
                    'num_layers': self.args.num_layers
                }
            }, model_path)
            print(f"  Saved agent_{i+1}_policy.pt")
        
        # Save critic networks
        for i, critic in enumerate(self.critics1):
            model_path = self.model_dir / f"agent_{i+1}_critic1.pt"
            torch.save({
                'model_state_dict': critic.state_dict(),
                'model_config': {
                    'num_states': self.num_states,
                    'num_actions': self.num_actions,
                    'n_obs_neighbors': self.n_obs_neighbors,
                    'num_ddl': self.num_ddl,
                    'model_type': self.args.model,
                    'hidden_size': self.args.hidden_size,
                    'num_layers': self.args.num_layers
                }
            }, model_path)
        
        for i, critic in enumerate(self.critics2):
            model_path = self.model_dir / f"agent_{i+1}_critic2.pt"
            torch.save({
                'model_state_dict': critic.state_dict(),
                'model_config': {
                    'num_states': self.num_states,
                    'num_actions': self.num_actions,
                    'n_obs_neighbors': self.n_obs_neighbors,
                    'num_ddl': self.num_ddl,
                    'model_type': self.args.model,
                    'hidden_size': self.args.hidden_size,
                    'num_layers': self.args.num_layers
                }
            }, model_path)
        
        # Save training metadata
        metadata = {
            'run_number': self.run_number,
            'model_id': self._generate_model_id(),
            'training_config': {
                'grid_size': self.args.grid_size,
                'pkg_p': self.args.pkg_p,
                'success_p': self.args.success_p,
                'ddl': self.args.ddl,
                'max_cycles': self.args.max_cycles,
                'n_obs_neighbors': self.args.n_obs_neighbors,
                'actor_lr': self.args.actor_lr,
                'critic_lr': self.args.critic_lr,
                'gamma': self.args.gamma,
                'tau': self.args.tau,
                'eta_mu': self.args.eta_mu,
                'rhs': self.args.rhs,
                'hidden_size': self.args.hidden_size,
                'num_layers': self.args.num_layers,
                'model': self.args.model
            },
            'environment_info': {
                'num_agents': self.num_agents,
                'num_actions': self.num_actions,
                'num_states': self.num_states,
                'grid_x': self.grid_x,
                'grid_y': self.grid_y,
                'num_ddl': self.num_ddl,
                'max_cycles': self.max_cycles
            },
            'training_stats': {
                'total_timesteps': self.args.total_timesteps,
                'final_episodic_return': self.log_data[-1]['avg_episodic_return'] if self.log_data else 0,
                'final_violation_rate': self.log_data[-1]['violation_rate'] if self.log_data else 0
            }
        }
        
        metadata_path = self.model_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved training_metadata.json")
        print(f"✅ All models saved successfully!")
        
        return self.run_number
    
    def _anneal_learning_rate(self, update):
        """Anneal learning rates."""
        frac = 1.0 - (update - 1.0) / self.args.total_timesteps
        ac_lrnow = frac * self.args.actor_lr
        cr_lrnow = frac * self.args.critic_lr
        
        for opt in self.actor_optimizer:
            opt.param_groups[0]["lr"] = ac_lrnow
        for opt in (self.critic1_optimizer + self.critic1_copy_optimizer + 
                   self.critic2_optimizer + self.critic2_copy_optimizer):
            opt.param_groups[0]["lr"] = cr_lrnow
    
    def _collect_trajectories(self):
        """Collect trajectories for training."""
        n_sample_traj = self.args.n_sample_traj
        warmup = self.args.total_timesteps // 3 * 2
        warmup_ratio = max(0, (warmup - self.args.total_timesteps) / max(1, warmup))
        
        # Get the actual observation size from environment by testing
        test_obs, _ = self.env_wrapper.reset()
        test_obs_batch = self.env_wrapper.get_batchified_obs(test_obs)
        obs_size = test_obs_batch.shape[1]  # Get actual observation size
        
        total_episodic_return = torch.zeros(n_sample_traj, self.num_agents).to(self.device)
        rb_obs = torch.zeros(n_sample_traj, self.max_cycles, self.num_agents, obs_size).to(self.device)
        rb_actions = torch.zeros(n_sample_traj, self.max_cycles, self.num_agents, dtype=torch.int64).to(self.device)
        rb_terms = torch.zeros(n_sample_traj, self.max_cycles, self.num_agents).to(self.device)
        rb_occupancy_measure = torch.zeros(n_sample_traj, self.num_agents, self.num_actions).to(self.device)
        rb_end_steps = torch.zeros(n_sample_traj, dtype=torch.long).to(self.device)
        
        for traj_id in range(n_sample_traj):
            next_obs, _ = self.env_wrapper.reset()
            
            for step in range(self.max_cycles):
                obs = self.env_wrapper.get_batchified_obs(next_obs)
                actions = self.env_wrapper.get_agent_actions(obs, self.actors)
                
                next_obs, rewards, terms, truncs, infos = self.env_wrapper.step(actions)
                
                rb_obs[traj_id, step] = obs
                rb_actions[traj_id, step] = actions
                rb_terms[traj_id, step] = self.env_wrapper.get_batchified_terms(terms)
                
                total_episodic_return[traj_id] += self.env_wrapper.get_batchified_rewards(rewards)
                rb_end_steps[traj_id] = step
                
                if self.env_wrapper.is_episode_done(terms, truncs):
                    break
            
            # Calculate occupancy measure for this trajectory
            rb_occupancy_measure[traj_id] = F.one_hot(
                rb_actions[traj_id, rb_end_steps[traj_id]], num_classes=int(self.num_actions)
            )
            for t in reversed(range(rb_end_steps[traj_id])):
                rb_occupancy_measure[traj_id] = F.one_hot(
                    rb_actions[traj_id, t], num_classes=int(self.num_actions)
                ) + self.args.gamma * rb_occupancy_measure[traj_id]
        
        return {
            'total_episodic_return': total_episodic_return,
            'rb_obs': rb_obs,
            'rb_actions': rb_actions,
            'rb_terms': rb_terms,
            'rb_occupancy_measure': rb_occupancy_measure,
            'rb_end_steps': rb_end_steps,
            'warmup_ratio': warmup_ratio
        }
    
    def _calculate_occupancy_measure(self, trajectories):
        """Calculate average occupancy measure."""
        avg_occupancy_measure = torch.mean(trajectories['rb_occupancy_measure'], dim=0)
        avg_occupancy_measure[:, 0] = 0  # Set no-action probability to 0
        return avg_occupancy_measure
    
    def _collect_q_data(self, avg_occupancy_measure):
        """Collect data for Q-function updates."""
        def r_mu(pos_y):
            select_occupancy = avg_occupancy_measure[torch.arange(pos_y.shape[0], device=self.device), pos_y]
            return ((1 - self.args.gamma) ** 2) * select_occupancy
        
        with torch.no_grad():
            # Reset environment once and collect data efficiently
            next_obs, _ = self.env_wrapper.reset()
            q_step = 0
            
            while q_step < self.n_sample_Q_steps:
                obs = self.env_wrapper.get_batchified_obs(next_obs)
                actions = self.env_wrapper.get_agent_actions(obs, self.actors)
                
                next_obs, rewards, terms, _, infos = self.env_wrapper.step(actions)
                
                # Check if episode is done
                if self.env_wrapper.is_episode_done(terms, None):
                    next_obs, _ = self.env_wrapper.reset()
                    continue
                
                # Store data
                self.rb_q_obs[q_step] = obs
                self.rb_q_obs_next[q_step] = self.env_wrapper.get_batchified_obs(next_obs)
                self.rb_q_actions[q_step] = actions
                self.rb_q_rewards1[q_step] = self.env_wrapper.get_batchified_rewards(rewards) / self.args.reward_scale
                self.rb_q_rewards2[q_step] = r_mu(actions)
                self.rb_q_terms[q_step] = self.env_wrapper.get_batchified_terms(terms)
                
                q_step += 1
    
    def _update_critics(self):
        """Update critic networks."""
        for agent_id in range(self.num_agents):
            # Get critic values
            neighbor_actions = gather_actions_2d(
                self.rb_q_actions, self.neighbor_dict, self.n_obs_neighbors, 
                self.num_actions, agent_id, self.device
            )
            neighbor_next_actions, agent_next_actions = gather_next_actions_2d(
                self.rb_q_obs_next, self.actors, self.neighbor_dict, self.n_obs_neighbors, 
                agent_id, self.num_actions, self.device
            )
            
            # Update critic1
            critic1_scores = get_all_scores(
                self.critics1, self.critics1_copy, self.critics1_target, self.critics1_copy_target,
                self.rb_q_obs[:, agent_id], self.rb_q_obs_next[:, agent_id],
                self.rb_q_actions[:, agent_id], agent_next_actions,
                agent_id, neighbor_actions, neighbor_next_actions
            )
            self._update_single_critic(
                critic1_scores, self.rb_q_rewards1[:, agent_id], self.rb_q_terms[:, agent_id],
                self.critics1[agent_id], self.critics1_copy[agent_id],
                self.critic1_optimizer[agent_id], self.critic1_copy_optimizer[agent_id],
                self.critics1_target[agent_id], self.critics1_copy_target[agent_id]
            )
            
            # Update critic2
            critic2_scores = get_all_scores(
                self.critics2, self.critics2_copy, self.critics2_target, self.critics2_copy_target,
                self.rb_q_obs[:, agent_id], self.rb_q_obs_next[:, agent_id],
                self.rb_q_actions[:, agent_id], agent_next_actions,
                agent_id, neighbor_actions, neighbor_next_actions
            )
            self._update_single_critic(
                critic2_scores, self.rb_q_rewards2[:, agent_id], self.rb_q_terms[:, agent_id],
                self.critics2[agent_id], self.critics2_copy[agent_id],
                self.critic2_optimizer[agent_id], self.critic2_copy_optimizer[agent_id],
                self.critics2_target[agent_id], self.critics2_copy_target[agent_id]
            )
    
    def _update_single_critic(self, scores, rewards, terms, critic, critic_copy, 
                             critic_optimizer, critic_copy_optimizer, critic_target, critic_copy_target):
        """Update a single critic network."""
        current_score, copy_current_score, next_score, copy_next_score = scores
        final_score = torch.minimum(next_score, copy_next_score)
        target_score = (1 - terms) * self.args.gamma * final_score + rewards
        
        # Update main critic
        loss_critic = F.mse_loss(current_score, target_score.detach())
        critic_optimizer.zero_grad(set_to_none=True)
        loss_critic.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), self.args.max_grad_norm)
        critic_optimizer.step()
        
        # Update copy critic
        loss_critic_copy = F.mse_loss(copy_current_score, target_score.detach())
        critic_copy_optimizer.zero_grad(set_to_none=True)
        loss_critic_copy.backward()
        nn.utils.clip_grad_norm_(critic_copy.parameters(), self.args.max_grad_norm)
        critic_copy_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(critic_copy.parameters(), critic_copy_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
    
    def _update_lagrangian(self, avg_occupancy_measure):
        """Update Lagrangian multiplier."""
        occupancy_entropy = 0.5 * ((1 - self.args.gamma) ** 2) * torch.sum(
            avg_occupancy_measure * avg_occupancy_measure, dim=1
        )
        total_violation = torch.sum(torch.clamp(-occupancy_entropy + self.args.rhs, min=0)).item()
        dual_mu = torch.clamp(-self.args.eta_mu * (occupancy_entropy - self.args.rhs), min=0)
        
        return dual_mu, total_violation
    
    def _update_actors(self, trajectories, dual_mu):
        """Update actor networks."""
        # Flatten trajectory data
        keep_mask = torch.arange(self.max_cycles).unsqueeze(0).repeat(self.args.n_sample_traj, 1).to(self.device)
        keep_mask = keep_mask <= trajectories['rb_end_steps'].unsqueeze(-1)
        b_obs = trajectories['rb_obs'][keep_mask]
        b_actions = trajectories['rb_actions'][keep_mask]
        
        for agent_id in range(self.num_agents):
            with torch.no_grad():
                # Get critic scores for this agent
                actor_neighbor_actions = gather_actions_2d(
                    b_actions, self.neighbor_dict, self.n_obs_neighbors, 
                    self.num_actions, agent_id, self.device
                )
                
                critic1_neighbor_score = gather_critic_score(
                    self.critics1, b_obs[:, agent_id], b_actions[:, agent_id], 
                    agent_id, actor_neighbor_actions
                )
                critic1_copy_neighbor_score = gather_critic_score(
                    self.critics1_copy, b_obs[:, agent_id], b_actions[:, agent_id], 
                    agent_id, actor_neighbor_actions
                )
                critic2_neighbor_score = gather_critic_score(
                    self.critics2, b_obs[:, agent_id], b_actions[:, agent_id], 
                    agent_id, actor_neighbor_actions
                )
                critic2_copy_neighbor_score = gather_critic_score(
                    self.critics2_copy, b_obs[:, agent_id], b_actions[:, agent_id], 
                    agent_id, actor_neighbor_actions
                )
                
                critic1_final_score = torch.minimum(critic1_neighbor_score, critic1_copy_neighbor_score)
                critic2_final_score = torch.minimum(critic2_neighbor_score, critic2_copy_neighbor_score)
                
                neighbor_critic_score1 = critic1_final_score + dual_mu[agent_id] * critic2_final_score
                neighbor_score_sum = neighbor_critic_score1.clone().detach()
                
                # Add neighbor scores
                for neighbor_id in self.neighbor_dict[agent_id]:
                    neighbor_actions = gather_actions_2d(
                        b_actions, self.neighbor_dict, self.n_obs_neighbors, 
                        self.num_actions, neighbor_id, self.device
                    )
                    
                    critic1_neighbor_score = gather_critic_score(
                        self.critics1, b_obs[:, neighbor_id], b_actions[:, neighbor_id], 
                        neighbor_id, neighbor_actions
                    )
                    critic1_copy_neighbor_score = gather_critic_score(
                        self.critics1_copy, b_obs[:, neighbor_id], b_actions[:, neighbor_id], 
                        neighbor_id, neighbor_actions
                    )
                    critic2_neighbor_score = gather_critic_score(
                        self.critics2, b_obs[:, neighbor_id], b_actions[:, neighbor_id], 
                        neighbor_id, neighbor_actions
                    )
                    critic2_copy_neighbor_score = gather_critic_score(
                        self.critics2_copy, b_obs[:, neighbor_id], b_actions[:, neighbor_id], 
                        neighbor_id, neighbor_actions
                    )
                    
                    critic1_final_score = torch.minimum(critic1_neighbor_score, critic1_copy_neighbor_score)
                    critic2_final_score = torch.minimum(critic2_neighbor_score, critic2_copy_neighbor_score)
                    
                    neighbor_critic_score2 = critic1_final_score + dual_mu[neighbor_id] * critic2_final_score
                    neighbor_score_sum = neighbor_score_sum + neighbor_critic_score2
                
                average_score = neighbor_score_sum / self.n_neighbor_dict[agent_id]
                average_score = (1 - trajectories['warmup_ratio']) * average_score + trajectories['warmup_ratio'] * neighbor_critic_score1
            
            # Update actor
            _, agent_logprob, _ = self.actors[agent_id].get_action(b_obs[:, agent_id], action=b_actions[:, agent_id])
            pg_loss = torch.mean(-average_score.detach() * agent_logprob)
            self.actor_optimizer[agent_id].zero_grad(set_to_none=True)
            pg_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.args.max_grad_norm)
            self.actor_optimizer[agent_id].step()
    
    def _log_results(self, update, trajectories, total_violation):
        """Log training results."""
        avg_episodic_return = trajectories['total_episodic_return'].mean().item()
        avg_end_step = trajectories['rb_end_steps'].to(torch.float32).mean().item()
        
        # Log to CSV
        if self.log_writer is not None:
            # Calculate gradient norms more efficiently
            actor_grads = [nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.args.max_grad_norm).item() 
                          for i in range(self.num_agents)]
            critic1_grads = [nn.utils.clip_grad_norm_(self.critics1[i].parameters(), self.args.max_grad_norm).item() 
                             for i in range(self.num_agents)]
            critic2_grads = [nn.utils.clip_grad_norm_(self.critics2[i].parameters(), self.args.max_grad_norm).item() 
                             for i in range(self.num_agents)]
            
            mean_actor_grad = sum(actor_grads) / self.num_agents
            mean_critic1_grad = sum(critic1_grads) / self.num_agents
            mean_critic2_grad = sum(critic2_grads) / self.num_agents
            
            self.log_writer.writerow([
                update, avg_episodic_return, total_violation, 
                mean_actor_grad, mean_critic1_grad, mean_critic2_grad
            ])
            # Also log to self.log_data for plotting/statistics
            self.log_data.append({
                "update": update,
                "avg_episodic_return": avg_episodic_return,
                "violation_rate": total_violation,
                "actor_grad_norm": mean_actor_grad,
                "critic1_grad_norm": mean_critic1_grad,
                "critic2_grad_norm": mean_critic2_grad
            })
    
    def _debug_visualization(self, update):
        """Create debug visualization."""
        with torch.no_grad():
            actions = torch.zeros(self.num_agents, dtype=torch.int64, device=self.device)
            action_probs = torch.zeros(self.num_agents, self.num_actions, device=self.device)
            total_test_episode_return = torch.zeros(self.num_agents, device=self.device)
            
            for sample_id in range(2):
                frames = [Image.fromarray(np.zeros_like(self.visual_background))]
                next_obs, _ = self.env_wrapper.reset()
                
                for step in range(self.max_cycles):
                    obs = self.env_wrapper.get_batchified_obs(next_obs)
                    actions, action_probs = self.env_wrapper.get_agent_actions_with_probs(obs, self.actors)
                    
                    next_obs, rewards, terms, truncs, infos = self.env_wrapper.step(
                        self.env_wrapper.get_unbatchified_actions(actions)
                    )
                    
                    obs_np = obs.data.cpu().numpy()
                    img = create_visualization_frame(
                        self.visual_background, actions, obs_np, rewards,
                        self.agent_shape, self.agent_gap, self.grid_y, self.n_obs_neighbors
                    )
                    frames.append(Image.fromarray(img))
                    
                    total_test_episode_return += self.env_wrapper.get_batchified_rewards(rewards)
                    
                    if self.env_wrapper.is_episode_done(terms, truncs):
                        break
                
                # Save GIF
                run_name = self._generate_run_name()
                frames[0].save(
                    f"./runs/{run_name}/step_{update}_sample_{sample_id}_return_{total_test_episode_return.mean().item():.3f}.gif",
                    save_all=True, append_images=frames[1:], optimize=False, duration=2000, loop=1
                )
    
    def cleanup(self):
        """Cleanup resources."""
        self.env_wrapper.close()
        if self.log_file is not None:
            self.log_file.close()
        if self.args.track:
            self.writer.close()
        
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Clear stored tensors to free memory
        if hasattr(self, 'rb_q_obs'):
            del self.rb_q_obs
        if hasattr(self, 'rb_q_obs_next'):
            del self.rb_q_obs_next
        if hasattr(self, 'rb_q_actions'):
            del self.rb_q_actions
        if hasattr(self, 'rb_q_rewards1'):
            del self.rb_q_rewards1
        if hasattr(self, 'rb_q_rewards2'):
            del self.rb_q_rewards2
        if hasattr(self, 'rb_q_terms'):
            del self.rb_q_terms 

    def save_hyperparameters(self):
        """Save hyperparameters to the run directory."""
        timestamp = name_with_datetime()
        filename = f"hyperparameters_{timestamp}.json"
        filepath = self.hyperparameters_dir / filename
        
        hyperparams = {
            'run_number': self.run_number,
            'timestamp': timestamp,
            'model_id': self._generate_model_id(),
            'hyperparameters': {
                'actor_lr': self.args.actor_lr,
                'critic_lr': self.args.critic_lr,
                'gamma': self.args.gamma,
                'tau': self.args.tau,
                'eta_mu': self.args.eta_mu,
                'rhs': self.args.rhs,
                'hidden_size': self.args.hidden_size,
                'num_layers': self.args.num_layers,
                'model': self.args.model,
                'total_timesteps': self.args.total_timesteps,
                'learning_starts': self.args.learning_starts,
                'batch_size': self.args.batch_size,
                'buffer_size': self.args.buffer_size,
                'target_network_frequency': self.args.target_network_frequency,
                'policy_frequency': self.args.policy_frequency,
                'noise_clip': self.args.noise_clip,
                'alpha': self.args.alpha,
                'autotune': self.args.autotune,
                'grad_norm': self.args.grad_norm,
                'target_policy_noise': self.args.target_policy_noise,
                'policy_noise': self.args.policy_noise,
                'learning_rate': self.args.learning_rate,
                'q_hidden_size': self.args.q_hidden_size,
                'p_hidden_size': self.args.p_hidden_size,
                'n_critics': self.args.n_critics,
                'use_sde': self.args.use_sde,
                'sde_sample_freq': self.args.sde_sample_freq,
                'use_sde_at_warmup': self.args.use_sde_at_warmup,
                'stats_window_size': self.args.stats_window_size,
                'tensorboard_log': self.args.tensorboard_log,
                'policy_kwargs': self.args.policy_kwargs,
                'verbose': self.args.verbose,
                'seed': self.args.seed,
                'deterministic': self.args.deterministic,
                'device': self.args.device,
                '_init_setup_model': self.args._init_setup_model,
                'wandb_project_name': self.args.wandb_project_name,
                'wandb_entity': self.args.wandb_entity,
                'capture_video': self.args.capture_video,
                'upload_model': self.args.upload_model,
                'hf_repo': self.args.hf_repo,
                'log_freq': self.args.log_freq,
                'save_freq': self.args.save_freq,
                'eval_freq': self.args.eval_freq,
                'num_envs': self.args.num_envs,
                'num_eval_episodes': self.args.num_eval_episodes,
                'optimize_memory_usage': self.args.optimize_memory_usage,
                'env_kwargs': self.args.env_kwargs,
                'eval_env_kwargs': self.args.eval_env_kwargs,
                'n_eval_envs': self.args.n_eval_envs,
                'dummy_vec_env': self.args.dummy_vec_env,
                'n_envs': self.args.n_envs,
                'monitor_dir': self.args.monitor_dir,
                'filename': self.args.filename,
                'vec_env_class': self.args.vec_env_class,
                'vec_env_kwargs': self.args.vec_env_kwargs,
                'monitor_kwargs': self.args.monitor_kwargs,
                'wrapper_class': self.args.wrapper_class,
                'env_wrapper': self.args.env_wrapper,
                'load_best_model_at_end': self.args.load_best_model_at_end,
                'auto_load_best_model': self.args.auto_load_best_model,
                'log_path': self.args.log_path,
                'save_dir': self.args.save_dir,
                'parameters_dir': self.args.parameters_dir,
                'grid_size': self.args.grid_size,
                'pkg_p': self.args.pkg_p,
                'success_p': self.args.success_p,
                'ddl': self.args.ddl,
                'max_cycles': self.args.max_cycles,
                'n_obs_neighbors': self.args.n_obs_neighbors,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        print(f"💾 Hyperparameters saved to: {filepath}")
        return filepath
    
    def save_environment_parameters(self):
        """Save environment parameters to the run directory."""
        timestamp = name_with_datetime()
        filename = f"environment_parameters_{timestamp}.json"
        filepath = self.parameters_dir / filename
        
        env_params = {
            'run_number': self.run_number,
            'timestamp': timestamp,
            'environment_config': {
                'grid_size': self.args.grid_size,
                'pkg_p': self.args.pkg_p,
                'success_p': self.args.success_p,
                'ddl': self.args.ddl,
                'max_cycles': self.args.max_cycles,
                'n_obs_neighbors': self.args.n_obs_neighbors,
                'grid_x': self.grid_x,
                'grid_y': self.grid_y,
                'num_agents': self.num_agents,
                'num_actions': self.num_actions,
                'num_states': self.num_states,
                'num_ddl': self.num_ddl,
                'max_cycles': self.max_cycles,
            },
            'environment_info': {
                'env_name': 'WirelessCommEnv',
                'env_type': 'MultiAgent',
                'observation_space': f"Box({self.num_states})",
                'action_space': f"Discrete({self.num_actions})",
                'num_agents': self.num_agents,
                'communication_range': self.args.n_obs_neighbors,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(env_params, f, indent=2)
        
        print(f"💾 Environment parameters saved to: {filepath}")
        return filepath
    
    def save_training_progress(self):
        """Save training progress data and figures."""
        timestamp = name_with_datetime()
        
        # Save training log CSV
        csv_filename = f"training_log_{timestamp}.csv"
        csv_filepath = self.training_progress_dir / csv_filename
        
        if self.log_data:
            import pandas as pd
            df = pd.DataFrame(self.log_data)
            df.to_csv(csv_filepath, index=False)
            print(f"💾 Training log saved to: {csv_filepath}")
        
        # Save training progress figures
        self._save_training_figures(timestamp)
        
        return csv_filepath
    
    def _save_training_figures(self, timestamp: str):
        """Save training progress figures."""
        if not self.log_data:
            return
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            df = pd.DataFrame(self.log_data)
            
            # Create training progress plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Run {self.run_number}', fontsize=16)
            
            # Episodic return
            axes[0, 0].plot(df['update'], df['avg_episodic_return'], 'b-', label='Average Return')
            axes[0, 0].set_title('Episodic Return')
            axes[0, 0].set_xlabel('Update Step')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Violation rate
            axes[0, 1].plot(df['update'], df['violation_rate'], 'r-', label='Violation Rate')
            axes[0, 1].set_title('Constraint Violation Rate')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Violation Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Loss values
            if 'actor_loss' in df.columns and 'critic_loss' in df.columns:
                axes[1, 0].plot(df['update'], df['actor_loss'], 'g-', label='Actor Loss')
                axes[1, 0].plot(df['update'], df['critic_loss'], 'm-', label='Critic Loss')
                axes[1, 0].set_title('Training Losses')
                axes[1, 0].set_xlabel('Update Step')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate
            if 'actor_lr' in df.columns and 'critic_lr' in df.columns:
                axes[1, 1].plot(df['update'], df['actor_lr'], 'c-', label='Actor LR')
                axes[1, 1].plot(df['update'], df['critic_lr'], 'y-', label='Critic LR')
                axes[1, 1].set_title('Learning Rates')
                axes[1, 1].set_xlabel('Update Step')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save figure
            fig_filename = f"training_progress_{timestamp}.png"
            fig_filepath = self.training_progress_dir / fig_filename
            plt.savefig(fig_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"💾 Training progress figure saved to: {fig_filepath}")
            
        except Exception as e:
            print(f"⚠️  Could not save training figures: {e}")
    
    def save_run_info(self):
        """Save additional run information."""
        timestamp = name_with_datetime()
        filename = f"run_info_{timestamp}.json"
        filepath = self.info_dir / filename
        
        run_info = {
            'run_number': self.run_number,
            'timestamp': timestamp,
            'model_id': self._generate_model_id(),
            'run_summary': {
                'total_timesteps': self.args.total_timesteps,
                'final_episodic_return': self.log_data[-1]['avg_episodic_return'] if self.log_data else 0,
                'final_violation_rate': self.log_data[-1]['violation_rate'] if self.log_data else 0,
                'training_duration': getattr(self, 'training_duration', 'Unknown'),
                'device_used': str(self.device),
                'num_agents': self.num_agents,
                'grid_size': f"{self.grid_x}x{self.grid_y}",
            },
            'system_info': {
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__,
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
            },
            'directory_structure': {
                'run_dir': str(self.run_dir),
                'parameters_dir': str(self.parameters_dir),
                'hyperparameters_dir': str(self.hyperparameters_dir),
                'model_dir': str(self.model_dir),
                'training_progress_dir': str(self.training_progress_dir),
                'execution_dir': str(self.execution_dir),
                'info_dir': str(self.info_dir),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"💾 Run info saved to: {filepath}")
        return filepath 