#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSDP Wireless Communication - Run and Test System

This script provides a comprehensive interface for running training and execution
with automatic run folder management (run_1, run_2, etc.) and organized result saving.

Author: Dongming Wang
Email: wdong025@ucr.edu
Project: DSDP
Folder: wireless_comm
Date: 2024
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import shutil
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dsdp_run_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DSDPRunManager:
    """Manages DSDP runs with automatic folder organization and result saving."""
    
    def __init__(self, base_dir: str = None, config_file: str = None):
        """
        Initialize the DSDP run manager.
        
        Args:
            base_dir: Base directory for the DSDP project
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
        # Core module paths
        self.trainer_module = "dsdp.wireless_comm.trainer"
        self.executor_module = "dsdp.wireless_comm.execute_model"
        self.config_module = "dsdp.wireless_comm.config"
        
        # Current run information
        self.current_run_number = None
        self.current_run_dir = None
        
        # Load configuration if provided
        self.config = self._load_config(config_file) if config_file else {}
        
        logger.info(f"DSDP Run Manager initialized at: {self.base_dir}")
        logger.info(f"Runs directory: {self.runs_dir}")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        if not config_file or not os.path.exists(config_file):
            return {}
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from: {config_file}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
            return {}
    
    def get_next_run_number(self) -> int:
        """Get the next available run number."""
        existing_runs = []
        for item in self.runs_dir.iterdir():
            if item.is_dir() and item.name.startswith('run_'):
                try:
                    run_num = int(item.name.split('_')[1])
                    existing_runs.append(run_num)
                except (ValueError, IndexError):
                    continue
        
        if not existing_runs:
            return 1
        else:
            return max(existing_runs) + 1
    
    def create_run_directory(self, run_number: int) -> Path:
        """Create a new run directory with proper structure."""
        run_dir = self.runs_dir / f"run_{run_number}"
        run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "parameters",
            "hyperparameters", 
            "model",
            "training_progress",
            "execution",
            "info"
        ]
        
        for subdir in subdirs:
            (run_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Created run directory: {run_dir}")
        return run_dir
    
    def save_run_info(self, run_dir: Path, run_type: str, config: Dict[str, Any]):
        """Save run information to the info directory."""
        info_file = run_dir / "info" / "run_info.json"
        
        run_info = {
            "run_number": int(run_dir.name.split('_')[1]),
            "run_type": run_type,
            "created_at": datetime.now().isoformat(),
            "config": config,
            "status": "created"
        }
        
        with open(info_file, 'w') as f:
            json.dump(run_info, f, indent=4)
        
        logger.info(f"Run info saved to: {info_file}")
    
    def run_training(self, config: Dict[str, Any] = None, run_number: int = None) -> Tuple[int, Path]:
        """
        Run training with automatic run folder management.
        
        Args:
            config: Training configuration dictionary
            run_number: Specific run number (None for auto-increment)
            
        Returns:
            Tuple of (run_number, run_directory)
        """
        # Determine run number
        if run_number is None:
            run_number = self.get_next_run_number()
        
        # Create run directory
        run_dir = self.create_run_directory(run_number)
        self.current_run_number = run_number
        self.current_run_dir = run_dir
        
        # Save run info
        self.save_run_info(run_dir, "training", config or {})
        
        logger.info(f"Starting training for run_{run_number}")
        logger.info(f"Run directory: {run_dir}")
        
        try:
            # Prepare training command
            cmd = [
                sys.executable, "-m", "dsdp.wireless_comm.main",
                "--run-dir", str(run_dir),
                "--save-dir", str(run_dir / "hyperparameters"),
                "--log-path", str(run_dir / "training_progress" / "training_log.csv")
            ]
            
            # Add configuration parameters
            if config:
                for key, value in config.items():
                    if key.startswith('--'):
                        cmd.extend([key, str(value)])
                    else:
                        # Convert underscores to hyphens for argument names
                        arg_name = key.replace('_', '-')
                        cmd.extend([f"--{arg_name}", str(value)])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Set PYTHONPATH to project root
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(self.base_dir).parent.parent)
            # Run training
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                text=True,
                timeout=36000,  # 10 hour timeout
                env=env
            )
            
            if result.returncode == 0:
                logger.info(f"Training completed successfully for run_{run_number}")
                self._update_run_status(run_dir, "completed")
                return run_number, run_dir
            else:
                logger.error(f"Training failed for run_{run_number}")
                logger.error(f"Error output: {result.stderr}")
                self._update_run_status(run_dir, "failed")
                raise RuntimeError(f"Training failed with return code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Training timed out for run_{run_number}")
            self._update_run_status(run_dir, "timeout")
            raise
        except Exception as e:
            logger.error(f"Training error for run_{run_number}: {e}")
            self._update_run_status(run_dir, "error")
            raise
    
    def run_execution(self, run_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run execution for a trained model.
        
        Args:
            run_path: Path to the run directory
            config: Execution configuration
            
        Returns:
            Execution results dictionary
        """
        run_dir = Path(run_path)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        
        logger.info(f"Starting execution for: {run_dir.name}")
        
        # Default execution config
        exec_config = {
            "num_episodes": 5,
            "max_steps": 50,
            "fps": 3,
            "save_animation": True,
            "save_frames": True
        }
        
        if config:
            exec_config.update(config)
        
        try:
            # Prepare execution command with improved parameters
            cmd = [
                sys.executable, "execute_model.py",
                "--run_path", str(run_dir),
                "--max_steps", str(exec_config["max_steps"]),
                "--fps", str(exec_config["fps"])
            ]
            
            # Add episode count if specified
            if exec_config.get("num_episodes", 1) > 1:
                cmd.extend(["--num_episodes", str(exec_config["num_episodes"])])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Set PYTHONPATH to project root
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(self.base_dir).parent.parent)
            
            # Run execution
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                text=True,
                timeout=1800,  # 30 minutes timeout
                env=env
            )
            
            if result.returncode == 0:
                logger.info(f"Execution completed successfully for {run_dir.name}")
                
                # Parse execution results
                results = self._parse_execution_results(run_dir, exec_config)
                
                # Save execution info
                self._save_execution_info(run_dir, exec_config, results)
                
                return results
            else:
                logger.error(f"Execution failed for {run_dir.name}")
                logger.error(f"Error output: {result.stderr}")
                raise RuntimeError(f"Execution failed with return code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Execution timed out for {run_dir.name}")
            raise
        except Exception as e:
            logger.error(f"Execution error for {run_dir.name}: {e}")
            raise
    
    def run_complete_pipeline(self, training_config: Dict[str, Any] = None, 
                            execution_config: Dict[str, Any] = None) -> Tuple[int, Path, Dict[str, Any]]:
        """
        Run complete pipeline: training + execution.
        
        Args:
            training_config: Training configuration
            execution_config: Execution configuration
            
        Returns:
            Tuple of (run_number, run_directory, execution_results)
        """
        logger.info("Starting complete pipeline: training + execution")
        
        # Run training
        run_number, run_dir = self.run_training(training_config)
        
        # Run execution
        execution_results = self.run_execution(str(run_dir), execution_config)
        
        logger.info(f"Complete pipeline finished for run_{run_number}")
        
        return run_number, run_dir, execution_results
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all available runs with their information."""
        runs = []
        
        for item in self.runs_dir.iterdir():
            if item.is_dir() and item.name.startswith('run_'):
                try:
                    run_num = int(item.name.split('_')[1])
                    info_file = item / "info" / "run_info.json"
                    
                    if info_file.exists():
                        with open(info_file, 'r') as f:
                            run_info = json.load(f)
                        runs.append(run_info)
                    else:
                        runs.append({
                            "run_number": run_num,
                            "run_type": "unknown",
                            "status": "no_info",
                            "created_at": "unknown"
                        })
                except (ValueError, IndexError):
                    continue
        
        # Sort by run number
        runs.sort(key=lambda x: x["run_number"])
        return runs
    
    def get_run_info(self, run_number: int) -> Dict[str, Any]:
        """Get detailed information about a specific run."""
        run_dir = self.runs_dir / f"run_{run_number}"
        info_file = run_dir / "info" / "run_info.json"
        
        if not info_file.exists():
            raise FileNotFoundError(f"Run info not found for run_{run_number}")
        
        with open(info_file, 'r') as f:
            run_info = json.load(f)
        
        # Add additional information
        run_info["run_directory"] = str(run_dir)
        run_info["has_model"] = (run_dir / "model").exists()
        run_info["has_training_progress"] = (run_dir / "training_progress").exists()
        run_info["has_execution"] = (run_dir / "execution").exists()
        
        return run_info
    
    def cleanup_run(self, run_number: int, keep_model: bool = True):
        """Clean up a run directory, optionally keeping the model."""
        run_dir = self.runs_dir / f"run_{run_number}"
        
        if not run_dir.exists():
            logger.warning(f"Run directory not found: {run_dir}")
            return
        
        logger.info(f"Cleaning up run_{run_number}")
        
        # Remove subdirectories
        subdirs_to_remove = ["training_progress", "execution", "info"]
        if not keep_model:
            subdirs_to_remove.append("model")
        
        for subdir in subdirs_to_remove:
            subdir_path = run_dir / subdir
            if subdir_path.exists():
                import shutil
                shutil.rmtree(subdir_path)
                logger.info(f"  Removed: {subdir}")
        
        # Update run info
        self._update_run_status(run_dir, "cleaned")
    
    def _update_run_status(self, run_dir: Path, status: str):
        """Update the status of a run."""
        info_file = run_dir / "info" / "run_info.json"
        
        if info_file.exists():
            with open(info_file, 'r') as f:
                run_info = json.load(f)
            
            run_info["status"] = status
            run_info["updated_at"] = datetime.now().isoformat()
            
            with open(info_file, 'w') as f:
                json.dump(run_info, f, indent=4)
    
    def _parse_execution_results(self, run_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse execution results from the run directory."""
        execution_dir = run_dir / "execution"
        
        results = {
            "run_number": int(run_dir.name.split('_')[1]),
            "execution_config": config,
            "executed_at": datetime.now().isoformat(),
            "files_created": []
        }
        
        if execution_dir.exists():
            # Count files
            for file_type in ["*.gif", "*.png", "*.mp4"]:
                files = list(execution_dir.glob(file_type))
                results["files_created"].extend([str(f.name) for f in files])
        
        return results
    
    def _save_execution_info(self, run_dir: Path, config: Dict[str, Any], results: Dict[str, Any]):
        """Save execution information."""
        exec_info_file = run_dir / "info" / "execution_info.json"
        
        with open(exec_info_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Execution info saved to: {exec_info_file}")
    
    def compare_runs(self, run_numbers: List[int]) -> Dict[str, Any]:
        """Compare multiple runs and generate comparison report."""
        if len(run_numbers) < 2:
            raise ValueError("At least 2 runs required for comparison")
        
        comparison = {
            "runs": [],
            "training_metrics": {},
            "execution_metrics": {},
            "configurations": {}
        }
        
        for run_num in run_numbers:
            try:
                run_info = self.get_run_info(run_num)
                comparison["runs"].append(run_info)
                
                # Load training progress if available
                training_log = run_info["run_directory"] + "/training_progress/training_log.csv"
                if os.path.exists(training_log):
                    df = pd.read_csv(training_log)
                    comparison["training_metrics"][f"run_{run_num}"] = {
                        "final_reward": df["reward"].iloc[-1] if "reward" in df.columns else None,
                        "total_steps": len(df),
                        "avg_reward": df["reward"].mean() if "reward" in df.columns else None
                    }
                
                # Load configuration
                if "config" in run_info:
                    comparison["configurations"][f"run_{run_num}"] = run_info["config"]
                    
            except Exception as e:
                logger.warning(f"Could not load run_{run_num}: {e}")
        
        return comparison
    
    def generate_comparison_plot(self, run_numbers: List[int], save_path: str = None) -> str:
        """Generate comparison plots for multiple runs."""
        comparison = self.compare_runs(run_numbers)
        
        if not comparison["training_metrics"]:
            raise ValueError("No training metrics available for comparison")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Run Comparison: {', '.join([f'run_{r}' for r in run_numbers])}")
        
        # Plot 1: Final rewards
        runs = list(comparison["training_metrics"].keys())
        final_rewards = [comparison["training_metrics"][r]["final_reward"] for r in runs]
        axes[0, 0].bar(runs, final_rewards)
        axes[0, 0].set_title("Final Rewards")
        axes[0, 0].set_ylabel("Reward")
        
        # Plot 2: Average rewards
        avg_rewards = [comparison["training_metrics"][r]["avg_reward"] for r in runs]
        axes[0, 1].bar(runs, avg_rewards)
        axes[0, 1].set_title("Average Rewards")
        axes[0, 1].set_ylabel("Reward")
        
        # Plot 3: Training progress (if available)
        for run_num in run_numbers:
            training_log = f"{self.runs_dir}/run_{run_num}/training_progress/training_log.csv"
            if os.path.exists(training_log):
                df = pd.read_csv(training_log)
                if "reward" in df.columns:
                    axes[1, 0].plot(df["reward"], label=f"run_{run_num}")
        
        axes[1, 0].set_title("Training Progress")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].legend()
        
        # Plot 4: Configuration comparison
        if comparison["configurations"]:
            config_keys = list(comparison["configurations"][runs[0]].keys())
            for i, key in enumerate(config_keys[:5]):  # Show first 5 configs
                values = [comparison["configurations"][r].get(key, "N/A") for r in runs]
                axes[1, 1].text(0.1, 0.9 - i*0.15, f"{key}: {values}", 
                               transform=axes[1, 1].transAxes, fontsize=10)
        
        axes[1, 1].set_title("Configuration Comparison")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = f"{self.base_dir}/plots/run_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to: {save_path}")
        return save_path
    
    def resume_training(self, run_number: int, additional_steps: int = 80000) -> Tuple[int, Path]:
        """Resume training from a previous run."""
        run_dir = self.runs_dir / f"run_{run_number}"
        
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: run_{run_number}")
        
        # Load previous configuration
        info_file = run_dir / "info" / "run_info.json"
        if not info_file.exists():
            raise FileNotFoundError(f"Run info not found for run_{run_number}")
        
        with open(info_file, 'r') as f:
            run_info = json.load(f)
        
        # Create new run for resumed training
        new_run_number = self.get_next_run_number()
        new_run_dir = self.create_run_directory(new_run_number)
        
        # Copy model files from previous run
        old_model_dir = run_dir / "model"
        new_model_dir = new_run_dir / "model"
        
        if old_model_dir.exists():
            shutil.copytree(old_model_dir, new_model_dir, dirs_exist_ok=True)
            logger.info(f"Copied model from run_{run_number} to run_{new_run_number}")
        
        # Update configuration for resumed training
        config = run_info.get("config", {}).copy()
        config["total_timesteps"] = additional_steps
        config["resume_from"] = f"run_{run_number}"
        
        # Save run info
        self.save_run_info(new_run_dir, "resumed_training", config)
        
        logger.info(f"Resuming training from run_{run_number} as run_{new_run_number}")
        
        # Run training
        return self.run_training(config, new_run_number)
    
    def batch_execute(self, run_numbers: List[int], config: Dict[str, Any] = None) -> Dict[int, Dict[str, Any]]:
        """Execute multiple runs in batch."""
        results = {}
        
        logger.info(f"Starting batch execution for {len(run_numbers)} runs")
        
        for run_num in run_numbers:
            try:
                run_path = str(self.runs_dir / f"run_{run_num}")
                result = self.run_execution(run_path, config)
                results[run_num] = result
                logger.info(f"Completed execution for run_{run_num}")
            except Exception as e:
                logger.error(f"Failed execution for run_{run_num}: {e}")
                results[run_num] = {"error": str(e)}
        
        return results
    
    def export_run(self, run_number: int, export_path: str = None) -> str:
        """Export a run to a compressed archive."""
        run_dir = self.runs_dir / f"run_{run_number}"
        
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: run_{run_number}")
        
        if export_path is None:
            export_path = f"{self.base_dir}/exports/run_{run_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Create tar.gz archive
        shutil.make_archive(
            export_path.replace('.tar.gz', ''),
            'gztar',
            run_dir
        )
        
        logger.info(f"Exported run_{run_number} to: {export_path}")
        return export_path
    
    def import_run(self, archive_path: str) -> int:
        """Import a run from a compressed archive."""
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        # Extract to temporary location
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(archive_path, temp_dir, 'gztar')
            
            # Find the run directory
            extracted_dirs = [d for d in os.listdir(temp_dir) if d.startswith('run_')]
            if not extracted_dirs:
                raise ValueError("No run directory found in archive")
            
            extracted_run_dir = os.path.join(temp_dir, extracted_dirs[0])
            
            # Get new run number
            new_run_number = self.get_next_run_number()
            new_run_dir = self.runs_dir / f"run_{new_run_number}"
            
            # Copy to runs directory
            shutil.copytree(extracted_run_dir, new_run_dir)
            
            # Update run info
            info_file = new_run_dir / "info" / "run_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    run_info = json.load(f)
                
                run_info["run_number"] = new_run_number
                run_info["imported_from"] = archive_path
                run_info["imported_at"] = datetime.now().isoformat()
                
                with open(info_file, 'w') as f:
                    json.dump(run_info, f, indent=4)
        
        logger.info(f"Imported run to: run_{new_run_number}")
        return new_run_number
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for the current environment."""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_usage": psutil.disk_usage(self.base_dir).percent,
            "dsdp_version": "1.0.0",  # This should be updated based on actual version
            "runs_count": len(self.list_runs()),
            "base_directory": str(self.base_dir)
        }
    
    def validate_run_directory(self, run_number: int) -> bool:
        """Validate that a run directory exists and has required files."""
        run_dir = self.runs_dir / f"run_{run_number}"
        
        if not run_dir.exists():
            return False
        
        # Check for required subdirectories
        required_dirs = ["info", "model"]
        for subdir in required_dirs:
            if not (run_dir / subdir).exists():
                return False
        
        # Check for run info file
        info_file = run_dir / "info" / "run_info.json"
        if not info_file.exists():
            return False
        
        return True
    
    def get_run_statistics(self) -> Dict[str, Any]:
        """Get statistics about all runs."""
        runs = self.list_runs()
        
        if not runs:
            return {"total_runs": 0}
        
        stats = {
            "total_runs": len(runs),
            "completed_runs": len([r for r in runs if r.get("status") == "completed"]),
            "failed_runs": len([r for r in runs if r.get("status") == "failed"]),
            "training_runs": len([r for r in runs if r.get("run_type") == "training"]),
            "resumed_runs": len([r for r in runs if r.get("run_type") == "resumed_training"]),
            "runs_with_models": 0,
            "runs_with_execution": 0,
            "oldest_run": None,
            "newest_run": None
        }
        
        # Count runs with models and execution
        for run in runs:
            run_dir = Path(run.get("run_directory", ""))
            if run_dir.exists():
                if (run_dir / "model").exists():
                    stats["runs_with_models"] += 1
                if (run_dir / "execution").exists():
                    stats["runs_with_execution"] += 1
        
        # Find oldest and newest runs
        timestamps = [r.get("created_at") for r in runs if r.get("created_at") != "unknown"]
        if timestamps:
            stats["oldest_run"] = min(timestamps)
            stats["newest_run"] = max(timestamps)
        
        return stats
    
    def cleanup_old_runs(self, days_old: int = 30, keep_completed: bool = True) -> List[int]:
        """Clean up runs older than specified days."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_runs = []
        
        for run in self.list_runs():
            if run.get("created_at") == "unknown":
                continue
            
            try:
                created_date = datetime.fromisoformat(run["created_at"])
                if created_date < cutoff_date:
                    # Skip completed runs if keep_completed is True
                    if keep_completed and run.get("status") == "completed":
                        continue
                    
                    run_number = run["run_number"]
                    self.cleanup_run(run_number, keep_model=False)
                    cleaned_runs.append(run_number)
                    logger.info(f"Cleaned up old run_{run_number}")
            except Exception as e:
                logger.warning(f"Could not process run_{run['run_number']}: {e}")
        
        return cleaned_runs
    
    def backup_runs(self, backup_dir: str = None) -> str:
        """Create a backup of all runs."""
        if backup_dir is None:
            backup_dir = f"{self.base_dir}/backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all run directories
        for run in self.list_runs():
            run_number = run["run_number"]
            source_dir = self.runs_dir / f"run_{run_number}"
            target_dir = backup_path / f"run_{run_number}"
            
            if source_dir.exists():
                shutil.copytree(source_dir, target_dir)
                logger.info(f"Backed up run_{run_number}")
        
        logger.info(f"Backup completed: {backup_path}")
        return str(backup_path)
    
    def create_default_config(self, config_path: str = None) -> str:
        """Create a default configuration file."""
        if config_path is None:
            config_path = f"{self.base_dir}/dsdp_config.json"
        
        default_config = {
            "training_configs": {
                "quick": {
                    "total_timesteps": 100,
                    "grid_size": 3,
                    "max_cycles": 8
                },
                "default": {
                    "total_timesteps": 1000,
                    "grid_size": 5,
                    "max_cycles": 12
                },
                "standard": {
                    "total_timesteps": 2000,
                    "grid_size": 5,
                    "max_cycles": 15
                },
                "extended": {
                    "total_timesteps": 5000,
                    "grid_size": 7,
                    "max_cycles": 20
                }
            },
            "execution_configs": {
                "quick": {"num_episodes": 2, "fps": 3},
                "default": {"num_episodes": 5, "fps": 3},
                "extended": {"num_episodes": 10, "fps": 5}
            },
            "system": {
                "default_timeout": 3600,
                "execution_timeout": 1800,
                "auto_backup": False,
                "backup_interval_days": 7
            }
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        logger.info(f"Default configuration created: {config_path}")
        return config_path


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="DSDP Wireless Communication - Run and Test System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                    Examples:
                    python run_and_test.py train                    # Run training with default config
                    python run_and_test.py train --config quick     # Run quick training
                    python run_and_test.py execute run_1            # Execute run_1
                    python run_and_test.py pipeline                 # Run complete pipeline
                    python run_and_test.py list                     # List all runs
                    python run_and_test.py info run_1               # Get run_1 info
                    python run_and_test.py cleanup run_1            # Cleanup run_1
                    python run_and_test.py compare 1 2 3 --plot     # Compare runs with plot
                    python run_and_test.py resume 1 --steps 2000    # Resume training from run_1
                    python run_and_test.py batch 1 2 3 --episodes 3 # Batch execute runs
                    python run_and_test.py export 1                 # Export run_1
                    python run_and_test.py import run_1.tar.gz      # Import run from archive
                    python run_and_test.py system                   # Show system information
                    python run_and_test.py stats                    # Show run statistics
                    python run_and_test.py backup                   # Create backup of all runs
                    python run_and_test.py cleanup-old --days 7     # Clean up runs older than 7 days
                    python run_and_test.py create-config            # Create default configuration file
                """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run training')
    train_parser.add_argument('--config', type=str, default='default',
                             choices=['default', 'quick', 'obs_neighbors_0', 'standard', 'extended'],
                             help='Training configuration')
    train_parser.add_argument('--run-number', type=int, help='Specific run number')
    
    # Execution command
    exec_parser = subparsers.add_parser('execute', help='Run execution')
    exec_parser.add_argument('run_id', type=str, help='Run number or path (e.g., 23, run_23, or runs/run_23)')
    exec_parser.add_argument('--fps', type=int, default=3, help='Animation FPS')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--training-config', type=str, default='default',
                                choices=['default', 'quick', 'standard', 'extended'],
                                help='Training configuration')
    pipeline_parser.add_argument('--execution-config', type=str, default='default',
                                choices=['default', 'quick', 'extended'],
                                help='Execution configuration')
    
    # List command
    subparsers.add_parser('list', help='List all runs')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get run information')
    info_parser.add_argument('run_number', type=int, help='Run number')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup run')
    cleanup_parser.add_argument('run_number', type=int, help='Run number')
    cleanup_parser.add_argument('--keep-model', action='store_true', help='Keep model files')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple runs')
    compare_parser.add_argument('run_numbers', type=int, nargs='+', help='Run numbers to compare')
    compare_parser.add_argument('--plot', action='store_true', help='Generate comparison plot')
    compare_parser.add_argument('--save-plot', type=str, help='Path to save comparison plot')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume training from a previous run')
    resume_parser.add_argument('run_number', type=int, help='Run number to resume from')
    resume_parser.add_argument('--steps', type=int, default=1000, help='Additional training steps')
    
    # Batch execute command
    batch_parser = subparsers.add_parser('batch', help='Execute multiple runs in batch')
    batch_parser.add_argument('run_numbers', type=int, nargs='+', help='Run numbers to execute')
    batch_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes per run')
    batch_parser.add_argument('--fps', type=int, default=3, help='Animation FPS')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export a run to archive')
    export_parser.add_argument('run_number', type=int, help='Run number to export')
    export_parser.add_argument('--path', type=str, help='Export path (optional)')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import a run from archive')
    import_parser.add_argument('archive_path', type=str, help='Path to archive file')
    
    # System info command
    subparsers.add_parser('system', help='Show system information')
    
    # Statistics command
    subparsers.add_parser('stats', help='Show run statistics')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create backup of all runs')
    backup_parser.add_argument('--path', type=str, help='Backup directory path')
    
    # Cleanup old command
    cleanup_old_parser = subparsers.add_parser('cleanup-old', help='Clean up old runs')
    cleanup_old_parser.add_argument('--days', type=int, default=30, help='Days old threshold')
    cleanup_old_parser.add_argument('--keep-completed', action='store_true', help='Keep completed runs')
    
    # Create config command
    create_config_parser = subparsers.add_parser('create-config', help='Create default configuration file')
    create_config_parser.add_argument('--path', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize run manager
    manager = DSDPRunManager()
    
    try:
        if args.command == 'train':
            # Training configurations
            configs = {
                'default': {
                    'total_timesteps': 100000,
                    'grid_size': 5,
                    'max_cycles': 30
                },
                'quick': {
                    'total_timesteps': 100,
                    'grid_size': 3,
                    'max_cycles': 8
                },
                'obs_neighbors_0': {
                    'total_timesteps': 100,
                    'grid_size': 3,
                    'max_cycles': 8,
                    'n_obs_neighbors': 0
                },
                'standard': {
                    'total_timesteps': 2000,
                    'grid_size': 5,
                    'max_cycles': 15
                },
                'extended': {
                    'total_timesteps': 5000,
                    'grid_size': 7,
                    'max_cycles': 20
                }
            }
            
            config = configs.get(args.config, configs['default'])
            run_number, run_dir = manager.run_training(config, args.run_number)
            
            print(f"\nTraining completed!")
            print(f"Run: run_{run_number}")
            print(f"Directory: {run_dir}")
            print(f"\nTo execute this model:")
            print(f"   python run_and_test.py execute run_{run_number}")
        
        elif args.command == 'execute':
            # Accept run_id as number, run_XX, or runs/run_XX
            run_id = args.run_id
            if run_id.isdigit():
                run_dir = manager.runs_dir / f"run_{run_id}"
            elif run_id.startswith('run_'):
                run_dir = manager.runs_dir / run_id
            elif run_id.startswith('runs/run_'):
                run_dir = Path(run_id)
            else:
                run_dir = Path(run_id)
            if not run_dir.exists():
                print(f"❌ Run directory not found: {run_dir}")
                return
            execution_config = {
                'num_episodes': 1,
                'max_steps': 50,
                'fps': args.fps
            }
            results = manager.run_execution(str(run_dir), execution_config)
            print(f"\nExecution completed!")
            print(f"Run: {run_dir}")
            print(f"Files created: {len(results.get('files_created', []))}")
        
        elif args.command == 'pipeline':
            
            # seed 
            # Training configurations
            training_configs = {
                'default': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30},
                'seed_73': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'seed': 73},
                'seed_100': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'seed': 100},
                'obs_neighbors_0_seed_1': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 0, 'seed': 1},
                'obs_neighbors_0_seed_73': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 0, 'seed': 73},
                'obs_neighbors_0_seed_100': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 0, 'seed': 100},
                'obs_neighbors_1_seed_1': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 1},
                'obs_neighbors_1_seed_73': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 1, 'seed': 73},
                'obs_neighbors_1_seed_100': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 1, 'seed': 100},
                'obs_neighbors_2_seed_1': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 2},
                'obs_neighbors_2_seed_73': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 2, 'seed': 73},
                'obs_neighbors_2_seed_100': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 2, 'seed': 100},
                'obs_neighbors_3_seed_1': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 3},
                'obs_neighbors_3_seed_73': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 3, 'seed': 73},
                'obs_neighbors_3_seed_100': {'total_timesteps': 80000, 'grid_size': 5, 'max_cycles': 30, 'n_obs_neighbors': 3, 'seed': 100},

                # 'standard': {'total_timesteps': 2000, 'grid_size': 5, 'max_cycles': 15},
                # 'extended': {'total_timesteps': 5000, 'grid_size': 7, 'max_cycles': 20}
            }
            
            # Execution configurations
            execution_configs = {
                'default': {'num_episodes': 5, 'max_steps': 50, 'fps': 3},
                'quick': {'num_episodes': 2, 'max_steps': 30, 'fps': 3},
                'extended': {'num_episodes': 10, 'max_steps': 100, 'fps': 5}
            }
            
            # training_config = training_configs.get(args.training_config, training_configs['default'])
            # execution_config = execution_configs.get(args.execution_config, execution_configs['default'])
            
            # run_number, run_dir, results = manager.run_complete_pipeline(training_config, execution_config)
            
            # print(f"\nComplete pipeline finished!")
            # print(f"Run: run_{run_number}")
            # print(f"Directory: {run_dir}")
            # print(f"Execution results: {len(results['files_created'])} files created")
            
            # # INSERT_YOUR_CODE
            # print("\nRunning complete pipeline for all training_configs in folder...")
            
            for config_name, training_config in training_configs.items():
                print(f"\n--- Running pipeline for config: {config_name} ---")
                # Use the same execution_config for all, or you could customize per config_name if desired
                training_config = training_configs.get(config_name, training_configs[config_name])
                execution_config = execution_configs.get(args.execution_config, execution_configs['default'])
                run_number, run_dir, results = manager.run_complete_pipeline(training_config, execution_config)
                print(f"Finished pipeline for {config_name}: run_{run_number}")
                print(f"Directory: {run_dir}")
                print(f"Execution results: {len(results.get('files_created', []))} files created")

            
        elif args.command == 'list':
            runs = manager.list_runs()
            
            if not runs:
                print("No runs found.")
            else:
                print(f"Found {len(runs)} runs:")
                print("-" * 80)
                print(f"{'Run':<6} {'Type':<12} {'Status':<12} {'Created':<20} {'Directory':<30}")
                print("-" * 80)
                
                for run in runs:
                    run_dir = f"run_{run['run_number']}"
                    created = run.get('created_at', 'unknown')[:19] if run.get('created_at') != 'unknown' else 'unknown'
                    print(f"{run['run_number']:<6} {run.get('run_type', 'unknown'):<12} {run.get('status', 'unknown'):<12} {created:<20} {run_dir:<30}")
        
        elif args.command == 'info':
            try:
                run_info = manager.get_run_info(args.run_number)
                
                print(f"\nRun {args.run_number} Information:")
                print("-" * 50)
                print(f"Type: {run_info.get('run_type', 'unknown')}")
                print(f"Status: {run_info.get('status', 'unknown')}")
                print(f"Created: {run_info.get('created_at', 'unknown')}")
                print(f"Directory: {run_info['run_directory']}")
                print(f"Has model: {run_info['has_model']}")
                print(f"Has training progress: {run_info['has_training_progress']}")
                print(f"Has execution: {run_info['has_execution']}")
                
                if 'config' in run_info and run_info['config']:
                    print(f"\nConfiguration:")
                    for key, value in run_info['config'].items():
                        print(f"  {key}: {value}")
                
            except FileNotFoundError as e:
                print(f"❌ Error: {e}")
        
        elif args.command == 'cleanup':
            manager.cleanup_run(args.run_number, args.keep_model)
            print(f"✅ Cleanup completed for run_{args.run_number}")
        
        elif args.command == 'compare':
            try:
                comparison = manager.compare_runs(args.run_numbers)
                
                print(f"\nRun Comparison: {', '.join([f'run_{r}' for r in args.run_numbers])}")
                print("-" * 60)
                
                # Print training metrics
                if comparison["training_metrics"]:
                    print("\nTraining Metrics:")
                    for run_key, metrics in comparison["training_metrics"].items():
                        print(f"  {run_key}:")
                        for metric, value in metrics.items():
                            print(f"    {metric}: {value}")
                
                # Generate plot if requested
                if args.plot:
                    plot_path = manager.generate_comparison_plot(
                        args.run_numbers, 
                        args.save_plot
                    )
                    print(f"\nComparison plot saved to: {plot_path}")
                
            except Exception as e:
                print(f"❌ Error during comparison: {e}")
        
        elif args.command == 'resume':
            try:
                run_number, run_dir = manager.resume_training(args.run_number, args.steps)
                print(f"\nTraining resumed!")
                print(f"New run: run_{run_number}")
                print(f"Directory: {run_dir}")
                print(f"Additional steps: {args.steps}")
            except Exception as e:
                print(f"❌ Error resuming training: {e}")
        
        elif args.command == 'batch':
            try:
                execution_config = {
                    'num_episodes': args.episodes,
                    'max_steps': 50,
                    'fps': args.fps
                }
                
                results = manager.batch_execute(args.run_numbers, execution_config)
                
                print(f"\nBatch execution completed!")
                print(f"Processed {len(results)} runs:")
                
                for run_num, result in results.items():
                    if "error" in result:
                        print(f"  ❌ run_{run_num}: {result['error']}")
                    else:
                        print(f"  ✅ run_{run_num}: {len(result.get('files_created', []))} files created")
                        if 'average_reward' in result:
                            print(f"     Average Reward: {result['average_reward']:.2f}")
                        
            except Exception as e:
                print(f"❌ Error during batch execution: {e}")
        
        elif args.command == 'export':
            try:
                export_path = manager.export_run(args.run_number, args.path)
                print(f"\nExport completed!")
                print(f"Run: run_{args.run_number}")
                print(f"Archive: {export_path}")
            except Exception as e:
                print(f"❌ Error during export: {e}")
        
        elif args.command == 'import':
            try:
                new_run_number = manager.import_run(args.archive_path)
                print(f"\nImport completed!")
                print(f"Archive: {args.archive_path}")
                print(f"New run: run_{new_run_number}")
            except Exception as e:
                print(f"❌ Error during import: {e}")
        
        elif args.command == 'system':
            try:
                system_info = manager.get_system_info()
                
                print(f"\nSystem Information:")
                print("-" * 50)
                print(f"Platform: {system_info['platform']}")
                print(f"Python: {system_info['python_version']}")
                print(f"CPU Cores: {system_info['cpu_count']}")
                print(f"Memory: {system_info['memory_total'] / (1024**3):.1f} GB")
                print(f"Disk Usage: {system_info['disk_usage']:.1f}%")
                print(f"DSDP Version: {system_info['dsdp_version']}")
                print(f"Total Runs: {system_info['runs_count']}")
                print(f"Base Directory: {system_info['base_directory']}")
                
            except Exception as e:
                print(f"❌ Error getting system info: {e}")
        
        elif args.command == 'stats':
            try:
                stats = manager.get_run_statistics()
                
                print(f"\nRun Statistics:")
                print("-" * 50)
                print(f"Total Runs: {stats['total_runs']}")
                print(f"Completed: {stats['completed_runs']}")
                print(f"Failed: {stats['failed_runs']}")
                print(f"Training Runs: {stats['training_runs']}")
                print(f"Resumed Runs: {stats['resumed_runs']}")
                print(f"With Models: {stats['runs_with_models']}")
                print(f"With Execution: {stats['runs_with_execution']}")
                
                if stats['oldest_run']:
                    print(f"Oldest Run: {stats['oldest_run'][:19]}")
                if stats['newest_run']:
                    print(f"Newest Run: {stats['newest_run'][:19]}")
                
            except Exception as e:
                print(f"❌ Error getting statistics: {e}")
        
        elif args.command == 'backup':
            try:
                backup_path = manager.backup_runs(args.path)
                print(f"\nBackup completed!")
                print(f"Backup location: {backup_path}")
            except Exception as e:
                print(f"❌ Error during backup: {e}")
        
        elif args.command == 'cleanup-old':
            try:
                cleaned_runs = manager.cleanup_old_runs(args.days, args.keep_completed)
                print(f"\nCleanup completed!")
                print(f"Cleaned {len(cleaned_runs)} old runs:")
                for run_num in cleaned_runs:
                    print(f"  - run_{run_num}")
            except Exception as e:
                print(f"❌ Error during cleanup: {e}")
        
        elif args.command == 'create-config':
            try:
                config_path = manager.create_default_config(args.path)
                print(f"\nDefault configuration created!")
                print(f"Config file: {config_path}")
                print(f"\nYou can now edit this file and use it with:")
                print(f"   python run_and_test.py --config {config_path} train")
            except Exception as e:
                print(f"❌ Error creating config: {e}")
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Error in {args.command}: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 