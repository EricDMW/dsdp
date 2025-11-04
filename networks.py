#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Networks for Wireless Communication Environment

This module provides Agent and Critic networks using the toolkit's neural networks.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from toolkit.neural_toolkit import MLPPolicyNetwork, MLPValueNetwork, NetworkUtils


class Agent(nn.Module):
    """Agent network for wireless communication environment."""
    
    def __init__(self, num_states, num_actions, num_ddl, model_type="mlp", 
                 hidden_size=256, num_layers=3):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_ddl = num_ddl
        self.model_type = model_type
        
        # Use toolkit's neural networks
        if model_type == "mlp":
            self.policy_network = MLPPolicyNetwork(
                input_dim=num_states,  # Use actual observation size
                output_dim=num_actions,
                hidden_dims=[hidden_size] * num_layers,
                activation="relu"
            )
        else:
            # Fallback to original implementation for other architectures
            self._create_original_network(hidden_size)
    
    def _create_original_network(self, hidden_size):
        """Create the original network architecture as fallback."""
        self.state_layer1 = self._layer_init(nn.Linear(self.num_states, 64))
        self.state_layer2 = self._layer_init(nn.Linear(64 * self.num_ddl, hidden_size))
        self.state_layer3 = self._layer_init(nn.Linear(hidden_size, 64))
        self.actor = self._layer_init(nn.Linear(64, self.num_actions), std=0.01)
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize layer weights."""
        torch.nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer, 'bias'):
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_action(self, x, action=None, return_prob=False):
        """Get action from the agent."""
        if self.model_type == "mlp":
            # Use toolkit's policy network
            state_latent = x  # Use observation directly, no flattening needed
            logits = self.policy_network(state_latent)
        else:
            # Use original implementation
            state_latent = torch.flatten(self.state_layer1(x), start_dim=1)
            logits = self.actor(F.relu(self.state_layer3(F.relu(self.state_layer2(F.relu(state_latent))))))
        
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        if return_prob:
            return action, probs.log_prob(action), probs.entropy(), probs.probs
        else:
            return action, probs.log_prob(action), probs.entropy()


class Critic(nn.Module):
    """Critic network for wireless communication environment."""
    
    def __init__(self, num_states, num_actions, n_obs_neighbors, num_ddl, 
                 model_type="mlp", hidden_size=256, num_layers=3):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.n_obs_neighbors = n_obs_neighbors
        self.num_ddl = num_ddl
        self.model_type = model_type
        
        # Calculate input dimensions - use actual observation size
        state_input_dim = num_states  # Use actual observation size
        # Calculate neighbor action input dimension
        neighbor_action_dim = max(1, (2 * n_obs_neighbors + 1) ** 2 - 1)  # Ensure at least 1 dimension
        action_input_dim = 8 * neighbor_action_dim  # Each neighbor action gets 8-dim embedding
        total_input_dim = state_input_dim + action_input_dim
        
        # Use toolkit's neural networks
        if model_type == "mlp":
            self.value_network = MLPValueNetwork(
                input_dim=total_input_dim,
                output_dim=num_actions,
                hidden_dims=[hidden_size] * num_layers,
                activation="relu"
            )
        else:
            # Fallback to original implementation
            self._create_original_network(hidden_size)
    
    def _create_original_network(self, hidden_size):
        """Create the original network architecture as fallback."""
        self.state_layer1 = self._layer_init(nn.Linear(self.num_states, 64))
        self.action_layer1 = self._layer_init(nn.Embedding(self.num_actions + 1, 8))
        # Ensure minimum dimension for neighbor actions
        neighbor_action_dim = max(1, (2 * self.n_obs_neighbors + 1) ** 2 - 1)
        self.critic = self._layer_init(nn.Linear(64 * self.num_ddl + 8 * neighbor_action_dim, hidden_size))
        self.critic1 = self._layer_init(nn.Linear(hidden_size, 64))
        self.critic2 = self._layer_init(nn.Linear(64, self.num_actions), std=1.0)
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize layer weights."""
        torch.nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer, 'bias'):
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_value(self, x, action):
        """Get value from the critic."""
        if self.model_type == "mlp":
            # Use toolkit's value network
            state_latent = x  # Use observation directly, no flattening needed
            
            # Handle neighbor actions - each action gets embedded
            # If no neighbors (action.shape[1] == 0), create a dummy embedding
            if action.shape[1] == 0:
                # Create a dummy action embedding when there are no neighbors
                batch_size = action.shape[0]
                device = action.device
                dummy_action = torch.full((batch_size, 1), self.num_actions, dtype=torch.long, device=device)
                action_emb = self.action_layer1(dummy_action[:, 0])
                action_latent = action_emb
            else:
                # Normal case: embed each neighbor action - more efficient
                # Pre-allocate tensor for better performance
                batch_size, num_neighbors = action.shape
                action_latent = torch.zeros(batch_size, 8 * num_neighbors, device=action.device)
                
                for i in range(num_neighbors):
                    start_idx = i * 8
                    end_idx = start_idx + 8
                    action_latent[:, start_idx:end_idx] = self.action_layer1(action[:, i])
            
            combined_input = torch.cat([state_latent, action_latent], dim=-1)
            return self.value_network(combined_input)
        else:
            # Use original implementation
            state_latent = torch.flatten(self.state_layer1(x), start_dim=1)
            action_latent = torch.flatten(self.action_layer1(action), start_dim=1)
            return self.critic2(F.relu(self.critic1(F.relu(self.critic(F.relu(torch.cat([state_latent, action_latent], dim=-1)))))))
    
    @property
    def action_layer1(self):
        """Get action embedding layer."""
        if not hasattr(self, '_action_layer1'):
            self._action_layer1 = self._layer_init(nn.Embedding(self.num_actions + 1, 8))
            # Move to the same device as the model
            if hasattr(self, 'value_network'):
                self._action_layer1 = self._action_layer1.to(next(self.value_network.parameters()).device)
            elif hasattr(self, 'critic'):
                self._action_layer1 = self._action_layer1.to(next(self.critic.parameters()).device)
        return self._action_layer1 