#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer with Loss Recording for Wireless Communication Environment

This module extends the base trainer to record Lagrangian multiplier errors
and policy convergence metrics during training.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .trainer import WirelessTrainer
from .utils import name_with_datetime


class WirelessTrainerWithLossRecord(WirelessTrainer):
    """
    Extended trainer that records Lagrangian multiplier errors and policy convergence.
    
    Usage:
        Phase 1 (Reference run): Train without reference_run_dir to establish baseline
        Phase 2 (Recording run): Train with reference_run_dir to record convergence metrics
    """
    
    def __init__(self, args, reference_run_dir=None):
        """
        Initialize the trainer with loss recording capability.
        
        Args:
            args: Configuration arguments
            reference_run_dir: Path to reference run directory (if None, this is the reference run)
        """
        super().__init__(args)
        
        self.reference_run_dir = reference_run_dir
        self.is_reference_run = reference_run_dir is None
        
        # Setup Lagrangian logging
        self.lagrangian_log_data = []
        self.lagrangian_log_file = None
        self.lagrangian_log_writer = None
        self.current_dual_mu = None  # Will be set during training
        
        # Setup theta error logging (2-norm and cosine)
        self.theta_2norm_log_data = []
        self.theta_2norm_log_file = None
        self.theta_2norm_log_writer = None
        
        self.theta_cosine_log_data = []
        self.theta_cosine_log_file = None
        self.theta_cosine_log_writer = None
        
        if not self.is_reference_run:
            # Load reference data from previous run
            print(f"\n📊 Loading reference data from: {self.reference_run_dir}")
            self.load_reference_data()
            self.setup_lagrangian_logging()
            self.setup_theta_2norm_logging()
            self.setup_theta_cosine_logging()
            
            # Verify initial policy error (should be high at start, then converge to 0 with same seed)
            print(f"\n🔍 Initial policy error check:")
            initial_2norm, initial_cosine = self._calculate_policy_metrics()
            print(f"   2-norm error: {initial_2norm.mean().item():.4f} ± {initial_2norm.std().item():.4f}")
            print(f"   Cosine similarity: {initial_cosine.mean().item():.6f} ± {initial_cosine.std().item():.6f}")
            print(f"   Expected with SAME seed: 2-norm → 0, cosine → 1.0")
            print(f"   Comparing X_k_i (current params) vs X_ref_i (constant reference)")
        else:
            print(f"\n📊 Running in REFERENCE mode - will save final Lagrangian and policies")
    
    def load_reference_data(self):
        """Load reference Lagrangian multipliers and policies from reference run."""
        reference_path = Path(self.reference_run_dir)
        
        # Load reference Lagrangian multipliers
        lagrangian_path = reference_path / "info" / "final_lagrangian.json"
        if not lagrangian_path.exists():
            raise FileNotFoundError(
                f"Reference Lagrangian file not found: {lagrangian_path}\n"
                f"Please run a reference training first without reference_run_dir parameter."
            )
        
        with open(lagrangian_path, 'r') as f:
            lagrangian_data = json.load(f)
            self.reference_lagrangian = torch.tensor(
                lagrangian_data['final_lagrangian_multipliers'],
                device=self.device
            )
        
        print(f"✅ Loaded reference Lagrangian multipliers: {self.reference_lagrangian.tolist()}")
        
        # Load reference policies as constant parameter tensors X_ref_i
        self.reference_policy_params = []  # Store X_ref_i as constant tensors
        model_dir = reference_path / "model"
        
        print(f"\n📋 Loading reference policy parameters as constant tensors:")
        
        # Load all reference policy parameters
        for i in range(self.num_agents):
            model_path = model_dir / f"agent_{i+1}_policy.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Reference policy not found: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract all parameters from the saved model
            state_dict = checkpoint['model_state_dict']
            
            # Flatten all parameters into a single tensor: X_ref_i
            with torch.no_grad():
                # Extract and flatten all parameters from the saved model
                ref_params = []
                for param_tensor in state_dict.values():
                    ref_params.append(param_tensor.flatten())
                
                # Concatenate all flattened parameters to get X_ref_i
                X_ref_i = torch.cat(ref_params)
            
            # Store as constant tensor (no gradient tracking, detached)
            self.reference_policy_params.append(X_ref_i)
            
            print(f"   Agent {i+1}: X_ref shape = {X_ref_i.shape}, total params = {X_ref_i.numel():,}")
        
        print(f"✅ Loaded {len(self.reference_policy_params)} reference policy parameter tensors")
        
        # Verify all agents have the same parameter dimension
        if len(self.reference_policy_params) > 1:
            first_dim = self.reference_policy_params[0].shape[0]
            all_same = all(X_ref.shape[0] == first_dim for X_ref in self.reference_policy_params)
            if all_same:
                print(f"   ✅ All agents have consistent parameter dimension: {first_dim:,}")
            else:
                print(f"   ⚠️  Warning: Different agents have different parameter dimensions")
    
    def setup_lagrangian_logging(self):
        """Setup CSV logging for Lagrangian errors and policy convergence."""
        log_path = self.training_progress_dir / "lagrangian.csv"
        self.lagrangian_log_file = open(str(log_path), mode='w', newline='')
        self.lagrangian_log_writer = csv.writer(self.lagrangian_log_file)
        
        # Header: update, agent_0_lagrangian_error, agent_1_lagrangian_error, ..., 
        #         agent_0_policy_cosine_sim, agent_1_policy_cosine_sim, ...,
        #         avg_lagrangian_error, avg_policy_cosine_sim
        header = ["update"]
        for i in range(self.num_agents):
            header.append(f"agent_{i}_lagrangian_error")
        for i in range(self.num_agents):
            header.append(f"agent_{i}_policy_cosine_sim")
        header.extend(["avg_lagrangian_error", "avg_policy_cosine_sim"])
        
        self.lagrangian_log_writer.writerow(header)
        print(f"📝 Lagrangian logging initialized: {log_path}")
    
    def setup_theta_2norm_logging(self):
        """Setup CSV logging for theta 2-norm error (Euclidean distance)."""
        log_path = self.training_progress_dir / "theta_error_2norm.csv"
        self.theta_2norm_log_file = open(str(log_path), mode='w', newline='')
        self.theta_2norm_log_writer = csv.writer(self.theta_2norm_log_file)
        
        # Header: update, agent_0_error, agent_1_error, ..., avg_error, std_error
        header = ["update"]
        for i in range(self.num_agents):
            header.append(f"agent_{i}_error_2norm")
        header.extend(["avg_error_2norm", "std_error_2norm"])
        
        self.theta_2norm_log_writer.writerow(header)
        print(f"📝 Theta 2-norm error logging initialized: {log_path}")
    
    def setup_theta_cosine_logging(self):
        """Setup CSV logging for theta cosine similarity."""
        log_path = self.training_progress_dir / "theta_error_cosine.csv"
        self.theta_cosine_log_file = open(str(log_path), mode='w', newline='')
        self.theta_cosine_log_writer = csv.writer(self.theta_cosine_log_file)
        
        # Header: update, agent_0_cosine, agent_1_cosine, ..., avg_cosine, std_cosine
        header = ["update"]
        for i in range(self.num_agents):
            header.append(f"agent_{i}_cosine")
        header.extend(["avg_cosine", "std_cosine"])
        
        self.theta_cosine_log_writer.writerow(header)
        print(f"📝 Theta cosine similarity logging initialized: {log_path}")
    
    def _calculate_lagrangian_error(self, current_dual_mu):
        """
        Calculate 1-norm error between current and reference Lagrangian multipliers.
        
        Args:
            current_dual_mu: Current Lagrangian multipliers [num_agents]
            
        Returns:
            errors: 1-norm error for each agent [num_agents]
        """
        # L1 norm (1-norm) for each agent
        # Use detach() to avoid gradient tracking issues
        with torch.no_grad():
            errors = torch.abs(current_dual_mu.detach() - self.reference_lagrangian)
        return errors
    
    def _calculate_policy_metrics(self):
        """
        Calculate both 2-norm error and cosine similarity between current and reference policies.
        
        At time step k, we compute:
        - 2-norm between X_k_i (current parameters) and X_ref_i (reference parameters)
        - Cosine similarity between X_k_i and X_ref_i
        
        Returns:
            error_2norms: 2-norm (Euclidean distance) for each agent [num_agents]
            cosine_sims: Cosine similarity for each agent [num_agents]
        """
        error_2norms = []
        cosine_sims = []
        
        for i in range(self.num_agents):
            # Get constant reference parameters X_ref_i (already loaded as tensor)
            X_ref_i = self.reference_policy_params[i]
            
            # Extract current parameters X_k_i from the training actor
            with torch.no_grad():  # CRITICAL: No gradients for metric calculation
                # Flatten all parameters from current policy
                X_k_i = torch.cat([
                    p.detach().clone().flatten() for p in self.actors[i].parameters()
                ])
            
            # Calculate 2-norm error ||X_k_i - X_ref_i||_2
            with torch.no_grad():
                error_2norm = torch.norm(X_k_i - X_ref_i, p=2).item()
                error_2norms.append(error_2norm)
                
                # Calculate cosine similarity between X_k_i and X_ref_i
                cosine_sim = F.cosine_similarity(
                    X_k_i.unsqueeze(0),
                    X_ref_i.unsqueeze(0),
                    dim=1
                ).item()
                cosine_sims.append(cosine_sim)
        
        return (torch.tensor(error_2norms, device=self.device), 
                torch.tensor(cosine_sims, device=self.device))
    
    def _update_lagrangian(self, avg_occupancy_measure):
        """
        Update Lagrangian multiplier and record errors if in recording mode.
        
        Args:
            avg_occupancy_measure: Average occupancy measure
            
        Returns:
            dual_mu: Updated Lagrangian multipliers
            total_violation: Total constraint violation
        """
        # Call parent method to update Lagrangian
        dual_mu, total_violation = super()._update_lagrangian(avg_occupancy_measure)
        
        # Store current Lagrangian (needed for both reference and recording runs)
        self.current_dual_mu = dual_mu.detach().clone()
        
        return dual_mu, total_violation
    
    def _log_results(self, update, trajectories, total_violation):
        """
        Log training results including Lagrangian errors and policy convergence.
        
        Args:
            update: Current update step
            trajectories: Trajectory data
            total_violation: Total constraint violation
        """
        # Call parent logging
        super()._log_results(update, trajectories, total_violation)
        
        # Additional logging for Lagrangian and policy convergence
        if not self.is_reference_run and self.lagrangian_log_writer is not None:
            # Calculate errors and metrics
            lagrangian_errors = self._calculate_lagrangian_error(self.current_dual_mu)
            policy_2norm_errors, policy_cosine_sims = self._calculate_policy_metrics()
            
            # Prepare log row for lagrangian.csv
            log_row = [update]
            
            # Add individual agent Lagrangian errors
            for i in range(self.num_agents):
                log_row.append(lagrangian_errors[i].item())
            
            # Add individual agent policy cosine similarities (kept for backward compatibility)
            for i in range(self.num_agents):
                log_row.append(policy_cosine_sims[i].item())
            
            # Add averages
            avg_lagrangian_error = lagrangian_errors.mean().item()
            avg_policy_similarity = policy_cosine_sims.mean().item()
            log_row.extend([avg_lagrangian_error, avg_policy_similarity])
            
            # Write to lagrangian.csv
            self.lagrangian_log_writer.writerow(log_row)
            
            # Store in memory for later use
            log_entry = {
                "update": update,
                "lagrangian_errors": lagrangian_errors.tolist(),
                "policy_similarities": policy_cosine_sims.tolist(),
                "avg_lagrangian_error": avg_lagrangian_error,
                "avg_policy_similarity": avg_policy_similarity
            }
            self.lagrangian_log_data.append(log_entry)
            
            # Log theta 2-norm errors to separate file
            if self.theta_2norm_log_writer is not None:
                theta_2norm_row = [update]
                for i in range(self.num_agents):
                    theta_2norm_row.append(policy_2norm_errors[i].item())
                theta_2norm_row.append(policy_2norm_errors.mean().item())  # avg_error_2norm
                theta_2norm_row.append(policy_2norm_errors.std().item())    # std_error_2norm
                self.theta_2norm_log_writer.writerow(theta_2norm_row)
                
                # Store 2-norm in memory
                theta_2norm_entry = {
                    "update": update,
                    "error_2norms": policy_2norm_errors.tolist(),
                    "avg_error_2norm": policy_2norm_errors.mean().item(),
                    "std_error_2norm": policy_2norm_errors.std().item()
                }
                self.theta_2norm_log_data.append(theta_2norm_entry)
            
            # Log theta cosine similarities to separate file
            if self.theta_cosine_log_writer is not None:
                theta_cosine_row = [update]
                for i in range(self.num_agents):
                    theta_cosine_row.append(policy_cosine_sims[i].item())
                theta_cosine_row.append(policy_cosine_sims.mean().item())  # avg_cosine
                theta_cosine_row.append(policy_cosine_sims.std().item())    # std_cosine
                self.theta_cosine_log_writer.writerow(theta_cosine_row)
                
                # Store cosine in memory
                theta_cosine_entry = {
                    "update": update,
                    "cosine_sims": policy_cosine_sims.tolist(),
                    "avg_cosine": policy_cosine_sims.mean().item(),
                    "std_cosine": policy_cosine_sims.std().item()
                }
                self.theta_cosine_log_data.append(theta_cosine_entry)
                
                # Print convergence progress every 50 updates
                if update % 50 == 0:
                    avg_lag_error = avg_lagrangian_error
                    avg_2norm = policy_2norm_errors.mean().item()
                    avg_cosine = policy_cosine_sims.mean().item()
                    print(f"\n   Update {update}: Lag Error = {avg_lag_error:.4f}, 2-norm = {avg_2norm:.4f}, Cosine = {avg_cosine:.6f}")
                    if avg_lag_error < 0.01 and avg_2norm < 1.0:
                        print(f"   ✅ Excellent convergence achieved!")
    
    def save_final_lagrangian(self):
        """Save final Lagrangian multipliers (only for reference run)."""
        if not self.is_reference_run:
            return
        
        if not hasattr(self, 'current_dual_mu'):
            print("⚠️  No Lagrangian data to save")
            return
        
        # Save final Lagrangian multipliers
        lagrangian_data = {
            'run_number': self.run_number,
            'timestamp': name_with_datetime(),
            'final_lagrangian_multipliers': self.current_dual_mu.tolist(),
            'num_agents': self.num_agents,
            'eta_mu': self.args.eta_mu,
            'rhs': self.args.rhs
        }
        
        filepath = self.info_dir / "final_lagrangian.json"
        with open(filepath, 'w') as f:
            json.dump(lagrangian_data, f, indent=2)
        
        print(f"💾 Final Lagrangian multipliers saved to: {filepath}")
        print(f"   Values: {self.current_dual_mu.tolist()}")
        return filepath
    
    def train(self):
        """Main training loop with loss recording."""
        if self.is_reference_run:
            print("\n🎯 PHASE 1: Reference training to establish baseline")
            print("   This run will save final Lagrangian multipliers and policies")
        else:
            print("\n🎯 PHASE 2: Training with loss recording")
            print("   This run will track convergence to reference values")
        
        # Call parent train method
        run_number = super().train()
        
        # Save final Lagrangian if this is reference run
        if self.is_reference_run:
            self.save_final_lagrangian()
            print(f"\n✅ Reference run completed!")
            print(f"   Use this run directory for the next phase:")
            print(f"   reference_run_dir = '{self.run_dir}'")
        else:
            # Print final convergence summary
            self._print_convergence_summary()
            
            # Save Lagrangian convergence figures
            self._save_lagrangian_figures()
            print(f"\n✅ Loss recording completed!")
        
        return run_number
    
    def _print_convergence_summary(self):
        """Print final convergence summary for Phase 2."""
        if not self.lagrangian_log_data or not self.theta_2norm_log_data or not self.theta_cosine_log_data:
            return
        
        print(f"\n{'='*70}")
        print(f"CONVERGENCE SUMMARY")
        print(f"{'='*70}")
        
        # Get initial and final values
        initial_lag = self.lagrangian_log_data[0]
        final_lag = self.lagrangian_log_data[-1]
        initial_2norm = self.theta_2norm_log_data[0]
        final_2norm = self.theta_2norm_log_data[-1]
        initial_cosine = self.theta_cosine_log_data[0]
        final_cosine = self.theta_cosine_log_data[-1]
        
        print(f"\n📊 Lagrangian Multiplier Error (L1-norm):")
        print(f"   Initial: {initial_lag['avg_lagrangian_error']:.4f}")
        print(f"   Final:   {final_lag['avg_lagrangian_error']:.4f}")
        
        if final_lag['avg_lagrangian_error'] < 0.01:
            print(f"   ✅ Excellent convergence (< 0.01)")
        elif final_lag['avg_lagrangian_error'] < 0.1:
            print(f"   ✓ Good convergence (< 0.1)")
        else:
            print(f"   ⚠️  Incomplete convergence (> 0.1)")
        
        print(f"\n📊 Policy Parameter Error (2-norm / Euclidean distance):")
        print(f"   Initial: {initial_2norm['avg_error_2norm']:.4f}")
        print(f"   Final:   {final_2norm['avg_error_2norm']:.4f}")
        reduction = (initial_2norm['avg_error_2norm'] - final_2norm['avg_error_2norm']) / initial_2norm['avg_error_2norm'] * 100
        print(f"   Reduction: {reduction:.1f}%")
        
        if final_2norm['avg_error_2norm'] < 1.0:
            print(f"   ✅ Excellent convergence (< 1.0)")
        elif final_2norm['avg_error_2norm'] < 10.0:
            print(f"   ✓ Good convergence (< 10.0)")
        elif final_2norm['avg_error_2norm'] < 50.0:
            print(f"   ⚠️  Moderate convergence (< 50.0)")
        else:
            print(f"   ⚠️  Poor convergence (≥ 50.0)")
            print(f"      💡 Tip: Verify both phases use the same seed!")
        
        print(f"\n📊 Policy Cosine Similarity:")
        print(f"   Initial: {initial_cosine['avg_cosine']:.6f}")
        print(f"   Final:   {final_cosine['avg_cosine']:.6f}")
        print(f"   Note: Cosine similarity is less informative; use 2-norm for convergence")
        
        print(f"\n📁 Output Files:")
        print(f"   • lagrangian.csv")
        print(f"   • theta_error_2norm.csv     (✅ Best for tracking convergence)")
        print(f"   • theta_error_cosine.csv")
        print(f"   • Shadow curve plots (PNG + PDF)")
        
        print(f"{'='*70}\n")
    
    def _save_lagrangian_figures(self):
        """Save Lagrangian error and policy convergence figures with shadow curves."""
        if not self.lagrangian_log_data:
            return
        
        try:
            # Use the dedicated shadow curve plotting script
            from .plot_lagrangian_shadow_curves import plot_lagrangian_shadow_curves
            
            # Find the lagrangian.csv file
            csv_path = self.training_progress_dir / "lagrangian.csv"
            
            if csv_path.exists():
                print(f"\n📊 Generating shadow curve plots...")
                plot_lagrangian_shadow_curves(
                    csv_path=str(csv_path),
                    output_dir=str(self.training_progress_dir),
                    smoothing_weight=0.9
                )
                print(f"✅ Shadow curve plots generated successfully!")
            else:
                print(f"⚠️  Could not find lagrangian.csv at {csv_path}")
            
        except Exception as e:
            print(f"⚠️  Could not save shadow curve figures: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup(self):
        """Cleanup resources including Lagrangian and theta log files."""
        # Close Lagrangian log file
        if self.lagrangian_log_file is not None:
            self.lagrangian_log_file.close()
        
        # Close theta 2-norm log file
        if self.theta_2norm_log_file is not None:
            self.theta_2norm_log_file.close()
        
        # Close theta cosine log file
        if self.theta_cosine_log_file is not None:
            self.theta_cosine_log_file.close()
        
        # Call parent cleanup
        super().cleanup()


def train_with_loss_record(args, reference_run_dir=None):
    """
    Convenience function to train with loss recording.
    
    Args:
        args: Configuration arguments
        reference_run_dir: Path to reference run directory (if None, this is the reference run)
        
    Returns:
        run_number: The run number of this training session
    """
    trainer = WirelessTrainerWithLossRecord(args, reference_run_dir)
    run_number = trainer.train()
    return run_number


if __name__ == "__main__":
    print("This module should be imported and used with appropriate configuration.")
    print("\nUsage example:")
    print("  # Phase 1: Reference run")
    print("  from dsdp.wireless_comm.trainer_with_loss_record import train_with_loss_record")
    print("  run_number = train_with_loss_record(args, reference_run_dir=None)")
    print("  ")
    print("  # Phase 2: Recording run")
    print("  reference_dir = f'dsdp/wireless_comm/runs/run_{run_number}'")
    print("  train_with_loss_record(args, reference_run_dir=reference_dir)")

