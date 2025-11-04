#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Wrapper for Wireless Communication

This module provides a wrapper for the existing wireless communication environment.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import numpy as np
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Decentralized-Safe-MARL-with-General-Utilities'))
from env_lib import WirelessCommEnv


class WirelessEnvironment:
    """Wrapper for the existing wireless communication environment."""
    
    def __init__(self, args, device):
        """
        Initialize the wireless communication environment.
        
        Args:
            args: Configuration arguments
            device: Device to run on (CPU/GPU)
        """
        self.args = args
        self.device = device
        
        # Environment parameters
        self.grid_x = args.grid_size
        self.grid_y = args.grid_size
        self.num_ddl = args.ddl
        self.pkg_p = args.pkg_p
        self.success_p = args.success_p
        self.n_obs_neighbors = args.n_obs_neighbors
        self.max_cycles = args.max_cycles
        
        # Create environment using existing WirelessCommEnv
        self._create_environment()
        
        # Get environment info
        self.num_agents = self.grid_x * self.grid_y
        self.num_actions = 5  # 0: idle, 1-4: transmit to different access points
        self.obs_size = (self.n_obs_neighbors * 2 + 1) ** 2
        self.num_states = self.obs_size
    
    def _create_environment(self):
        """Create the wireless communication environment using existing WirelessCommEnv."""
        self.env = WirelessCommEnv(
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            ddl=self.num_ddl,
            render_mode="rgb_array",
            max_iter=self.max_cycles,
            save_gif=False,
            gif_path="",
            debug_info=True,
            n_obs_neighbors=self.n_obs_neighbors,
            packet_arrival_probability=self.pkg_p,
            success_transmission_probability=self.success_p
        )
    
    def reset(self):
        """Reset the environment."""
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, actions):
        """Take a step in the environment."""
        next_obs, rewards, terms, truncs, infos = self.env.step(actions)
        return next_obs, rewards, terms, truncs, infos
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_batchified_obs(self, obs):
        """Convert observations to batch format."""
        if isinstance(obs, dict):
            # Convert dict to batch - more efficient list comprehension
            obs_list = [obs[a] for a in obs]
            return torch.stack(obs_list, dim=0).to(self.device)
        else:
            # Handle numpy arrays and other formats
            if isinstance(obs, np.ndarray):
                return torch.from_numpy(obs).to(self.device)
            elif hasattr(obs, 'device') and obs.device != self.device:
                return obs.to(self.device)
            return obs
    
    def get_unbatchified_actions(self, actions):
        """Convert actions to unbatch format."""
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if hasattr(self.env, 'possible_agents'):
            return {a: actions[i] for i, a in enumerate(self.env.possible_agents)}
        else:
            return actions
    
    def get_batchified_rewards(self, rewards):
        """Convert rewards to batch format."""
        if isinstance(rewards, dict):
            # Convert dict to batch - more efficient
            rewards_list = [rewards[a] for a in rewards]
            return torch.tensor(rewards_list, device=self.device, dtype=torch.float32)
        else:
            # Already in batch format, just ensure it's on the right device
            if hasattr(rewards, 'device') and rewards.device != self.device:
                return rewards.to(self.device)
            return rewards
    
    def get_batchified_terms(self, terms):
        """Convert termination flags to batch format."""
        if isinstance(terms, dict):
            # Convert dict to batch - more efficient
            terms_list = [terms[a] for a in terms]
            return torch.tensor(terms_list, device=self.device, dtype=torch.bool)
        else:
            # Already in batch format, just ensure it's on the right device
            if hasattr(terms, 'device') and terms.device != self.device:
                return terms.to(self.device)
            return terms
    
    def get_agent_actions(self, obs, actors):
        """Get actions from all agents."""
        # Pre-allocate tensor for better performance
        actions = torch.zeros(self.num_agents, dtype=torch.int64, device=self.device)
        
        # Collect actions more efficiently
        for i in range(self.num_agents):
            actions[i] = actors[i].get_action(obs[i].unsqueeze(0))[0]
        
        return actions
    
    def get_agent_actions_with_probs(self, obs, actors):
        """Get actions and probabilities from all agents."""
        actions = torch.zeros(self.num_agents, dtype=torch.int64, device=self.device)
        action_probs = torch.zeros(self.num_agents, self.num_actions, device=self.device)
        
        for agent_id in range(self.num_agents):
            agent_action, _, _, agent_probs = actors[agent_id].get_action(
                obs[agent_id].unsqueeze(0), return_prob=True
            )
            actions[agent_id] = agent_action
            action_probs[agent_id] = agent_probs
        
        return actions, action_probs
    
    def is_episode_done(self, terms, truncs):
        """Check if episode is done."""
        # Handle both dictionary and boolean return types
        if isinstance(terms, dict):
            return any([terms[a] for a in terms]) or any([truncs[a] for a in truncs])
        else:
            # Handle boolean or single value
            return bool(terms) or bool(truncs)
    
    def get_environment_info(self):
        """Get environment information."""
        return {
            'num_agents': self.num_agents,
            'num_actions': self.num_actions,
            'num_states': self.num_states,
            'obs_size': self.obs_size,
            'grid_x': self.grid_x,
            'grid_y': self.grid_y,
            'num_ddl': self.num_ddl,
            'n_obs_neighbors': self.n_obs_neighbors,
            'max_cycles': self.max_cycles
        }
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def start_frame_collection(self):
        """Start collecting frames for GIF creation."""
        if hasattr(self.env, 'start_frame_collection'):
            self.env.start_frame_collection()
    
    def stop_frame_collection(self):
        """Stop collecting frames."""
        if hasattr(self.env, 'stop_frame_collection'):
            self.env.stop_frame_collection()
    
    def save_gif(self, path, fps=3):
        """Save collected frames as GIF."""
        if hasattr(self.env, 'save_gif'):
            self.env.save_gif(path, fps=fps) 