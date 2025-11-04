#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for Wireless Communication Environment

This module provides utility functions for the wireless communication environment.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import cv2
import numpy as np
import torch
from collections import defaultdict
from PIL import Image, ImageDraw
from datetime import datetime
import pytz


def name_with_datetime():
    """Generate a name with current datetime."""
    now = datetime.now(tz=pytz.utc)
    now = now.astimezone(pytz.timezone('US/Pacific'))
    return now.strftime("%Y-%m-%d_%H:%M:%S")


def find_neighbors(grid_x, grid_y, n_obs_neighbors):
    """
    Find neighbors for each agent in the grid.
    
    Args:
        grid_x: Grid width
        grid_y: Grid height
        n_obs_neighbors: Number of neighbors to observe
        
    Returns:
        neighbor_dict: Dictionary mapping agent_id to neighbor_ids
        n_neighbor_dict: Dictionary mapping agent_id to number of neighbors
    """
    neighbor_dict = defaultdict(list)
    n_neighbor_dict = defaultdict(int)
    
    for i in range(grid_x):
        for j in range(grid_y):
            agent_id = i * grid_y + j
            
            # Handle the case when n_obs_neighbors = 0
            if n_obs_neighbors == 0:
                # No neighbors to observe, but we need at least one placeholder
                # to maintain consistent tensor dimensions
                neighbor_dict[agent_id].append(-1)
                n_neighbor_dict[agent_id] = 1
            else:
                # Normal case: find neighbors within the observation range
                for l in range(i - n_obs_neighbors, i + n_obs_neighbors + 1):
                    for m in range(j - n_obs_neighbors, j + n_obs_neighbors + 1):
                        if l < 0 or m < 0 or l >= grid_x or m >= grid_y:
                            neighbor_dict[agent_id].append(-1)
                        else:
                            neighbor_dict[agent_id].append(l * grid_y + m)
                            n_neighbor_dict[agent_id] += 1
                
                # Remove the agent itself from neighbor list
                if agent_id in neighbor_dict[agent_id]:
                    neighbor_dict[agent_id].remove(agent_id)
                    n_neighbor_dict[agent_id] = max(1, n_neighbor_dict[agent_id] - 1)
    
    return neighbor_dict, n_neighbor_dict


def image_background(grid_x, grid_y, agent_shape, agent_gap):
    """
    Create background image for visualization.
    
    Args:
        grid_x: Grid width
        grid_y: Grid height
        agent_shape: Size of agent representation
        agent_gap: Gap between agents
        
    Returns:
        Background image as numpy array
    """
    small_gap = (agent_gap - agent_shape) // 2
    w = (grid_y + 2) * agent_shape + (grid_y - 1) * agent_gap
    h = (grid_x + 2) * agent_shape + (grid_x - 1) * agent_gap
    im = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    
    # Draw agent rectangles
    for i in range(grid_x):
        for j in range(grid_y):
            draw.rectangle([
                agent_shape + j * (agent_shape + agent_gap),
                agent_shape + i * (agent_shape + agent_gap),
                agent_shape * 2 + j * (agent_shape + agent_gap),
                agent_shape * 2 + i * (agent_shape + agent_gap)
            ], fill=(141, 219, 252), outline='black')
    
    # Draw connection points
    for i in range(grid_x - 1):
        for j in range(grid_y - 1):
            leftUpPoint = (
                agent_shape + agent_shape * (j + 1) + j * agent_gap + small_gap,
                agent_shape + agent_shape * (i + 1) + i * agent_gap + small_gap
            )
            rightDownPoint = (
                leftUpPoint[0] + agent_shape,
                leftUpPoint[1] + agent_shape
            )
            draw.ellipse([leftUpPoint, rightDownPoint], fill=(167, 252, 141), outline="black")
    
    return np.array(im)


def create_visualization_frame(background_img, actions, obs_np, rewards, agent_shape, agent_gap, grid_y, n_obs_neighbors):
    """
    Create a visualization frame for the environment.
    
    Args:
        background_img: Background image
        actions: Actions taken by agents
        obs_np: Observations as numpy array
        rewards: Rewards received by agents
        agent_shape: Size of agent representation
        agent_gap: Gap between agents
        grid_y: Grid height
        n_obs_neighbors: Number of neighbors to observe
        
    Returns:
        Visualization frame as numpy array
    """
    img = background_img.copy()
    reward_list = list(rewards.values())
    
    for agent_id in range(len(actions)):
        # Add observation text
        text_start_x = agent_shape * (1 + agent_id % grid_y) + agent_gap * (agent_id % grid_y)
        text_start_y = agent_shape * (2 + agent_id // grid_y) + agent_gap * (agent_id // grid_y)
        cv2.putText(
            img,
            f"{obs_np[agent_id, :, n_obs_neighbors * (2 * n_obs_neighbors + 1) + n_obs_neighbors].tolist()}",
            (text_start_x, text_start_y + 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0)
        )
        
        # Skip if no action (action == 0)
        if actions[agent_id] == 0:
            continue
        
        # Draw action arrows
        start_x = agent_shape * (1.5 + agent_id % grid_y) + agent_gap * (agent_id % grid_y)
        start_y = agent_shape * (1.5 + agent_id // grid_y) + agent_gap * (agent_id // grid_y)
        arrow_color = reward_list[agent_id] == 1
        
        # Define arrow positions based on action
        if actions[agent_id] == 1:  # Up
            arrow_start_x, arrow_start_y = start_x - agent_shape // 2, start_y - agent_shape // 2
            arrow_end_x, arrow_end_y = arrow_start_x - agent_gap // 2, arrow_start_y - agent_gap // 2
        elif actions[agent_id] == 2:  # Down
            arrow_start_x, arrow_start_y = start_x - agent_shape // 2, start_y + agent_shape // 2
            arrow_end_x, arrow_end_y = arrow_start_x - agent_gap // 2, arrow_start_y + agent_gap // 2
        elif actions[agent_id] == 3:  # Left
            arrow_start_x, arrow_start_y = start_x + agent_shape // 2, start_y - agent_shape // 2
            arrow_end_x, arrow_end_y = arrow_start_x + agent_gap // 2, arrow_start_y - agent_gap // 2
        elif actions[agent_id] == 4:  # Right
            arrow_start_x, arrow_start_y = start_x + agent_shape // 2, start_y + agent_shape // 2
            arrow_end_x, arrow_end_y = arrow_start_x + agent_gap // 2, arrow_start_y + agent_gap // 2
        
        # Draw the arrow
        img = cv2.arrowedLine(
            img, 
            (int(arrow_start_x), int(arrow_start_y)),
            (int(arrow_end_x), int(arrow_end_y)),
            (246, 123, 210) if arrow_color else (0, 0, 0), 2
        )
    
    return img


def calculate_occupancy_measure(actions, end_steps, num_actions, gamma):
    """
    Calculate occupancy measure for trajectories.
    
    Args:
        actions: Actions taken in trajectories
        end_steps: End steps for each trajectory
        num_actions: Number of possible actions
        gamma: Discount factor
        
    Returns:
        Occupancy measure tensor
    """
    import torch.nn.functional as F
    
    occupancy_measure = F.one_hot(actions, num_classes=int(num_actions))
    
    for t in reversed(range(end_steps)):
        occupancy_measure = F.one_hot(actions[t], num_classes=int(num_actions)) + gamma * occupancy_measure
    
    return occupancy_measure


def calculate_entropy_reward(actions, avg_occupancy_measure):
    """
    Calculate entropy-based reward for constraint satisfaction.
    
    Args:
        actions: Actions taken by agents
        avg_occupancy_measure: Average occupancy measure
        
    Returns:
        Entropy reward tensor
    """
    select_occupancy = avg_occupancy_measure[torch.arange(actions.shape[0], device=actions.device), actions]
    return ((1 - 0.7) ** 2) * select_occupancy  # Using gamma=0.7 as default


# Utility functions from the original utils module
def gather_actions_2d(b_actions, neighbor_dict, n_obs_neighbors, num_actions, agent_id, device):
    """Gather actions from neighbors in 2D grid."""
    # Calculate neighbor action dimension, ensuring at least 1 dimension
    neighbor_action_dim = max(1, (2 * n_obs_neighbors + 1) ** 2 - 1)
    
    # Pre-allocate tensor and fill with empty actions
    neighbor_actions = torch.full((b_actions.shape[0], neighbor_action_dim), 
                                 int(num_actions), dtype=torch.int64, device=device)
    
    # If there are no neighbors, just return the tensor filled with empty actions
    if n_obs_neighbors == 0:
        return neighbor_actions
    
    # Normal case: gather actions from neighbors
    for i, neighbor_id in enumerate(neighbor_dict[agent_id]):
        if neighbor_id != -1:
            neighbor_actions[:, i] = b_actions[:, neighbor_id]
    
    return neighbor_actions


def gather_next_actions_2d(b_obs_next, actors, neighbor_dict, n_obs_neighbors, agent_id, num_actions, device):
    """Gather next actions from neighbors in 2D grid."""
    agent_next_actions, _, _ = actors[agent_id].get_action(b_obs_next[:, agent_id])
    
    # Calculate neighbor action dimension, ensuring at least 1 dimension
    neighbor_action_dim = max(1, (2 * n_obs_neighbors + 1) ** 2 - 1)
    
    # Pre-allocate tensor and fill with empty actions
    neighbor_next_actions = torch.full((b_obs_next.shape[0], neighbor_action_dim), 
                                      int(num_actions), dtype=torch.int64, device=device)
    
    # If there are no neighbors, just return the tensor filled with empty actions
    if n_obs_neighbors == 0:
        return neighbor_next_actions, agent_next_actions
    
    # Normal case: gather next actions from neighbors
    for i, neighbor_id in enumerate(neighbor_dict[agent_id]):
        if neighbor_id != -1:
            neighbor_next_actions[:, i], _, _ = actors[neighbor_id].get_action(b_obs_next[:, neighbor_id])
    
    return neighbor_next_actions, agent_next_actions


def gather_critic_score(critics, b_obs, agent_actions, agent_id, neighbor_actions):
    """Gather critic score for a specific agent."""
    critic_score_all = critics[agent_id].get_value(b_obs, neighbor_actions)
    critic_score = critic_score_all.gather(1, agent_actions.unsqueeze(-1)).squeeze()
    return critic_score


def get_all_scores(critics, critics_copy, critics_target, critics_copy_target, b_obs, b_obs_next, agent_actions,
                   agent_actions_next, agent_id, neighbor_actions, neighbor_actions_next):
    """Get all critic scores for an agent."""
    current_score = gather_critic_score(critics, b_obs, agent_actions, agent_id, neighbor_actions)
    copy_current_score = gather_critic_score(critics_copy, b_obs, agent_actions, agent_id, neighbor_actions)
    next_score = gather_critic_score(critics_target, b_obs_next, agent_actions_next, agent_id, neighbor_actions_next)
    copy_next_score = gather_critic_score(critics_copy_target, b_obs_next, agent_actions_next, agent_id,
                                          neighbor_actions_next)
    return current_score, copy_current_score, next_score, copy_next_score 