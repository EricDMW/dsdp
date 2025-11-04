#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSDP Wireless Communication Module

This module provides a complete training and execution framework for the DSDP algorithm
on wireless communication environments.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Date: 2024
"""

from .config import get_args, save_hyperparameters, load_hyperparameters
from .environment import WirelessEnvironment
from .networks import Agent, Critic
from .trainer import WirelessTrainer
from .execute_model import ModelExecutor
from .utils import name_with_datetime, find_neighbors, gather_critic_score
from .visualization import TrainingVisualizer

__all__ = [
    # Core components
    'WirelessEnvironment',
    'Agent',
    'Critic',
    'WirelessTrainer',
    'ModelExecutor',
    
    # Configuration
    'get_args',
    'save_hyperparameters',
    'load_hyperparameters',
    
    # Utilities
    'name_with_datetime',
    'find_neighbors',
    'gather_critic_score',
    
    # Visualization
    'TrainingVisualizer',
]

__version__ = "1.0.0"
__author__ = "Dongming Wang"
__email__ = "wdong025@ucr.edu" 