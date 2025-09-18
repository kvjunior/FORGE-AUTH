"""
FORGE-AUTH: Utilities, Data Handling, and Visualization
=======================================================

This module provides comprehensive utility functions, data handling capabilities,
and visualization tools for the FORGE-AUTH authentication system. It includes
data generation for experiments, preprocessing pipelines, performance monitoring,
and visualization utilities essential for system operation and evaluation.

Key Components:
- Synthetic data generation for temporal transaction graphs
- Data preprocessing and feature engineering
- Performance monitoring and profiling utilities
- Visualization tools for analysis and reporting
- System configuration management
- Logging and debugging utilities
- Statistical analysis helpers
- Cryptographic utility functions

Author: FORGE-AUTH Research Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, TemporalData
from torch_geometric.utils import to_networkx, from_networkx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TSNE
import logging
import json
import yaml
import pickle
import time
import os
import sys
import hashlib
import random
import warnings
import psutil
import GPUtil
import tracemalloc
from datetime import datetime, timedelta
import threading
import queue
from pathlib import Path
import subprocess
import platform
import socket
import getpass
from contextlib import contextmanager
import functools
import itertools
from scipy import stats
from scipy.sparse import csr_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility constants
UTILS_CONSTANTS = {
    'RANDOM_SEED': 42,
    'EPSILON': 1e-8,
    'MAX_RETRIES': 3,
    'CACHE_SIZE': 1000,
    'FIGURE_DPI': 300,
    'FIGURE_SIZE': (10, 6),
    'COLOR_PALETTE': 'viridis',
    'TIMESTAMP_FORMAT': '%Y-%m-%d %H:%M:%S',
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

def set_random_seeds(seed: int = UTILS_CONSTANTS['RANDOM_SEED']):
    """
    Set random seeds for reproducibility across all libraries.
    
    This function ensures consistent random number generation across NumPy,
    Python's random module, and PyTorch, which is essential for reproducible
    experiments and debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seeds set to {seed}")

def create_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger instance with optional file output.
    
    This function creates a logger with both console and file handlers,
    using a consistent format across the FORGE-AUTH system.
    """
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger_instance.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(UTILS_CONSTANTS['LOG_FORMAT'])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)
    
    return logger_instance

class Timer:
    """
    Context manager for timing code execution with detailed statistics.
    
    This class provides precise timing measurements and can be used to profile
    different parts of the FORGE-AUTH system for performance optimization.
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"{self.name} completed in {self.duration:.4f} seconds")
        else:
            self.logger.error(f"{self.name} failed after {self.duration:.4f} seconds")
        
        return False

class MemoryTracker:
    """
    Tracks memory usage for monitoring resource consumption.
    
    This class monitors both CPU and GPU memory usage, helping to identify
    memory leaks and optimize resource utilization in the authentication system.
    """
    
    def __init__(self, max_memory_gb: float = 32.0):
        self.max_memory_gb = max_memory_gb
        self.memory_history = []
        self.start_snapshot = None
        
        # Start memory tracing
        tracemalloc.start()
        self.start_snapshot = tracemalloc.take_snapshot()
    
    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage and return statistics."""
        memory_stats = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_stats['cpu_memory_gb'] = memory_info.rss / (1024 ** 3)
        memory_stats['cpu_memory_percent'] = process.memory_percent()
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = []
            for gpu in GPUtil.getGPUs():
                gpu_memory.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used_gb': gpu.memoryUsed / 1024,
                    'memory_total_gb': gpu.memoryTotal / 1024,
                    'memory_percent': gpu.memoryUtil * 100
                })
            memory_stats['gpu_memory'] = gpu_memory
        
        # Check against limit
        if memory_stats['cpu_memory_gb'] > self.max_memory_gb:
            logger.warning(f"Memory usage ({memory_stats['cpu_memory_gb']:.2f} GB) "
                         f"exceeds limit ({self.max_memory_gb} GB)")
        
        # Store in history
        memory_stats['timestamp'] = time.time()
        self.memory_history.append(memory_stats)
        
        return memory_stats
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """Get detailed memory profiling information."""
        current_snapshot = tracemalloc.take_snapshot()
        top_stats = current_snapshot.compare_to(self.start_snapshot, 'lineno')
        
        profile = {
            'top_allocations': [],
            'total_allocated_mb': sum(stat.size_diff for stat in top_stats) / (1024 ** 2)
        }
        
        for stat in top_stats[:10]:
            profile['top_allocations'].append({
                'file': stat.traceback.format()[0],
                'size_diff_mb': stat.size_diff / (1024 ** 2),
                'count_diff': stat.count_diff
            })
        
        return profile

class DataGenerator:
    """
    Generates synthetic temporal transaction graph data for experiments.
    
    This class creates realistic synthetic data that mimics financial transaction
    patterns, including normal behavior and various types of fraudulent activities.
    """
    
    def __init__(self, num_users: int = 10000, num_transactions: int = 1000000,
                 fraud_rate: float = 0.01, temporal_span_days: int = 30):
        self.num_users = num_users
        self.num_transactions = num_transactions
        self.fraud_rate = fraud_rate
        self.temporal_span = temporal_span_days * 24 * 3600  # Convert to seconds
        
        # User profiles for realistic behavior
        self.user_profiles = self._generate_user_profiles()
        
        # Transaction patterns
        self.transaction_patterns = self._define_transaction_patterns()
        
        logger.info(f"Initialized DataGenerator with {num_users} users, "
                   f"{num_transactions} transactions, {fraud_rate:.2%} fraud rate")
    
    def _generate_user_profiles(self) -> pd.DataFrame:
        """Generate realistic user profiles with behavioral characteristics."""
        profiles = []
        
        for user_id in range(self.num_users):
            profile = {
                'user_id': user_id,
                'account_age_days': np.random.exponential(365),
                'activity_level': np.random.choice(['low', 'medium', 'high'], 
                                                 p=[0.6, 0.3, 0.1]),
                'typical_amount_range': np.random.choice(['small', 'medium', 'large'],
                                                       p=[0.7, 0.25, 0.05]),
                'risk_score': np.random.beta(2, 5),  # Skewed towards low risk
                'preferred_hours': np.random.choice(['morning', 'afternoon', 'evening', 'night'],
                                                  p=[0.3, 0.4, 0.25, 0.05])
            }
            
            # Add behavioral consistency
            profile['behavioral_consistency'] = np.random.beta(5, 2)  # Skewed towards consistent
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def _define_transaction_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define various transaction patterns including fraudulent ones."""
        patterns = {
            'normal_small': {
                'amount_range': (10, 100),
                'frequency': 'high',
                'time_pattern': 'regular',
                'merchant_diversity': 'high'
            },
            'normal_medium': {
                'amount_range': (100, 1000),
                'frequency': 'medium',
                'time_pattern': 'regular',
                'merchant_diversity': 'medium'
            },
            'normal_large': {
                'amount_range': (1000, 10000),
                'frequency': 'low',
                'time_pattern': 'business_hours',
                'merchant_diversity': 'low'
            },
            'fraud_card_testing': {
                'amount_range': (0.01, 10),
                'frequency': 'burst',
                'time_pattern': 'random',
                'merchant_diversity': 'single'
            },
            'fraud_account_takeover': {
                'amount_range': (500, 5000),
                'frequency': 'burst',
                'time_pattern': 'unusual',
                'merchant_diversity': 'high'
            },
            'fraud_synthetic_identity': {
                'amount_range': (100, 2000),
                'frequency': 'gradual_increase',
                'time_pattern': 'evolving',
                'merchant_diversity': 'medium'
            }
        }
        
        return patterns
    
    def generate_temporal_graph(self) -> List[TemporalData]:
        """Generate temporal transaction graph data."""
        logger.info("Generating temporal transaction graph data")
        
        transactions = []
        edge_index = []
        edge_attr = []
        edge_times = []
        labels = []
        
        # Generate normal transactions
        num_normal = int(self.num_transactions * (1 - self.fraud_rate))
        normal_transactions = self._generate_normal_transactions(num_normal)
        
        # Generate fraudulent transactions
        num_fraud = self.num_transactions - num_normal
        fraud_transactions = self._generate_fraud_transactions(num_fraud)
        
        # Combine and shuffle
        all_transactions = normal_transactions + fraud_transactions
        random.shuffle(all_transactions)
        
        # Convert to graph format
        for transaction in all_transactions:
            # Edge from sender to receiver
            edge_index.append([transaction['sender'], transaction['receiver']])
            
            # Edge attributes
            edge_attr.append([
                transaction['amount'],
                transaction['hour_of_day'],
                transaction['day_of_week'],
                transaction['merchant_category'],
                transaction['location_risk'],
                transaction['velocity_score'],
                transaction['device_fingerprint'],
                transaction['behavioral_score']
            ])
            
            # Timestamp
            edge_times.append(transaction['timestamp'])
            
            # Label (0: normal, 1: fraud)
            labels.append(transaction['is_fraud'])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_times = torch.tensor(edge_times, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Create node features
        node_features = self._create_node_features()
        
        # Create temporal data objects
        temporal_data = TemporalData(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_time=edge_times,
            y=labels
        )
        
        # Split into time windows for temporal processing
        window_size = 3600  # 1 hour windows
        temporal_windows = self._split_temporal_windows(temporal_data, window_size)
        
        logger.info(f"Generated {len(temporal_windows)} temporal windows")
        
        return temporal_windows
    
    def _generate_normal_transactions(self, num_transactions: int) -> List[Dict[str, Any]]:
        """Generate normal transaction patterns."""
        transactions = []
        
        for _ in range(num_transactions):
            # Select random user
            user_profile = self.user_profiles.sample(1).iloc[0]
            
            # Generate transaction based on user profile
            transaction = self._generate_transaction_from_profile(user_profile, is_fraud=False)
            transactions.append(transaction)
        
        return transactions
    
    def _generate_fraud_transactions(self, num_transactions: int) -> List[Dict[str, Any]]:
        """Generate fraudulent transaction patterns."""
        transactions = []
        fraud_types = ['card_testing', 'account_takeover', 'synthetic_identity']
        
        for _ in range(num_transactions):
            fraud_type = np.random.choice(fraud_types)
            
            if fraud_type == 'card_testing':
                transaction = self._generate_card_testing_fraud()
            elif fraud_type == 'account_takeover':
                transaction = self._generate_account_takeover_fraud()
            else:
                transaction = self._generate_synthetic_identity_fraud()
            
            transactions.append(transaction)
        
        return transactions
    
    def _generate_transaction_from_profile(self, user_profile: pd.Series,
                                         is_fraud: bool = False) -> Dict[str, Any]:
        """Generate transaction based on user profile."""
        # Determine amount based on profile
        if user_profile['typical_amount_range'] == 'small':
            amount = np.random.lognormal(3, 1)
        elif user_profile['typical_amount_range'] == 'medium':
            amount = np.random.lognormal(5, 1.5)
        else:
            amount = np.random.lognormal(7, 2)
        
        # Time patterns
        if user_profile['preferred_hours'] == 'morning':
            hour = np.random.randint(6, 12)
        elif user_profile['preferred_hours'] == 'afternoon':
            hour = np.random.randint(12, 18)
        elif user_profile['preferred_hours'] == 'evening':
            hour = np.random.randint(18, 23)
        else:
            hour = np.random.randint(0, 6)
        
        # Generate timestamp
        timestamp = np.random.uniform(0, self.temporal_span)
        
        # Create transaction
        transaction = {
            'sender': int(user_profile['user_id']),
            'receiver': np.random.randint(0, self.num_users),
            'amount': amount,
            'timestamp': timestamp,
            'hour_of_day': hour,
            'day_of_week': int(timestamp / 86400) % 7,
            'merchant_category': np.random.randint(0, 20),
            'location_risk': user_profile['risk_score'] * (1 + is_fraud * 2),
            'velocity_score': self._calculate_velocity_score(user_profile, is_fraud),
            'device_fingerprint': hash(str(user_profile['user_id'])) % 1000,
            'behavioral_score': user_profile['behavioral_consistency'] * (1 - is_fraud * 0.5),
            'is_fraud': int(is_fraud)
        }
        
        return transaction
    
    def _generate_card_testing_fraud(self) -> Dict[str, Any]:
        """Generate card testing fraud pattern."""
        base_user = np.random.randint(0, self.num_users)
        
        return {
            'sender': base_user,
            'receiver': np.random.randint(0, self.num_users),
            'amount': np.random.uniform(0.01, 10),
            'timestamp': np.random.uniform(0, self.temporal_span),
            'hour_of_day': np.random.randint(0, 24),
            'day_of_week': np.random.randint(0, 7),
            'merchant_category': 99,  # Suspicious category
            'location_risk': np.random.uniform(0.7, 1.0),
            'velocity_score': np.random.uniform(0.8, 1.0),
            'device_fingerprint': np.random.randint(5000, 10000),  # Unknown device
            'behavioral_score': np.random.uniform(0, 0.3),
            'is_fraud': 1
        }
    
    def _generate_account_takeover_fraud(self) -> Dict[str, Any]:
        """Generate account takeover fraud pattern."""
        compromised_user = np.random.randint(0, self.num_users)
        
        return {
            'sender': compromised_user,
            'receiver': np.random.randint(0, self.num_users),
            'amount': np.random.uniform(500, 5000),
            'timestamp': np.random.uniform(0, self.temporal_span),
            'hour_of_day': np.random.choice([2, 3, 4, 22, 23]),  # Unusual hours
            'day_of_week': np.random.randint(0, 7),
            'merchant_category': np.random.choice([15, 16, 17]),  # High-risk categories
            'location_risk': np.random.uniform(0.6, 0.9),
            'velocity_score': 0.9,  # High velocity
            'device_fingerprint': np.random.randint(10000, 20000),  # New device
            'behavioral_score': 0.1,  # Very different behavior
            'is_fraud': 1
        }
    
    def _generate_synthetic_identity_fraud(self) -> Dict[str, Any]:
        """Generate synthetic identity fraud pattern."""
        synthetic_user = self.num_users + np.random.randint(0, 1000)
        
        return {
            'sender': synthetic_user,
            'receiver': np.random.randint(0, self.num_users),
            'amount': np.random.uniform(100, 2000),
            'timestamp': np.random.uniform(0, self.temporal_span),
            'hour_of_day': np.random.randint(9, 17),  # Business hours
            'day_of_week': np.random.randint(1, 6),  # Weekdays
            'merchant_category': np.random.randint(0, 20),
            'location_risk': 0.5,  # Medium risk
            'velocity_score': 0.3,  # Gradual buildup
            'device_fingerprint': hash(str(synthetic_user)) % 1000,
            'behavioral_score': 0.6,  # Somewhat consistent
            'is_fraud': 1
        }
    
    def _calculate_velocity_score(self, user_profile: pd.Series, is_fraud: bool) -> float:
        """Calculate transaction velocity score."""
        base_velocity = {
            'low': 0.2,
            'medium': 0.4,
            'high': 0.6
        }[user_profile['activity_level']]
        
        if is_fraud:
            return min(1.0, base_velocity * np.random.uniform(1.5, 3.0))
        else:
            return base_velocity * np.random.uniform(0.8, 1.2)
    
    def _create_node_features(self) -> torch.Tensor:
        """Create node features for users."""
        node_features = []
        
        for _, profile in self.user_profiles.iterrows():
            features = [
                profile['account_age_days'] / 365,  # Normalized
                {'low': 0.2, 'medium': 0.5, 'high': 0.8}[profile['activity_level']],
                {'small': 0.2, 'medium': 0.5, 'large': 0.8}[profile['typical_amount_range']],
                profile['risk_score'],
                profile['behavioral_consistency']
            ]
            
            node_features.append(features)
        
        return torch.tensor(node_features, dtype=torch.float)
    
    def _split_temporal_windows(self, data: TemporalData, 
                              window_size: float) -> List[Data]:
        """Split temporal data into time windows."""
        windows = []
        
        # Sort by time
        sorted_indices = torch.argsort(data.edge_time)
        
        # Group into windows
        current_window_start = 0
        window_edges = []
        window_attrs = []
        window_times = []
        window_labels = []
        
        for idx in sorted_indices:
            edge_time = data.edge_time[idx].item()
            
            if edge_time - current_window_start > window_size:
                # Create window data
                if window_edges:
                    window_data = Data(
                        x=data.x,
                        edge_index=torch.stack(window_edges, dim=1),
                        edge_attr=torch.stack(window_attrs),
                        edge_time=torch.tensor(window_times),
                        y=torch.tensor(window_labels)
                    )
                    windows.append(window_data)
                
                # Reset for new window
                current_window_start = edge_time
                window_edges = []
                window_attrs = []
                window_times = []
                window_labels = []
            
            # Add to current window
            window_edges.append(data.edge_index[:, idx])
            window_attrs.append(data.edge_attr[idx])
            window_times.append(edge_time)
            window_labels.append(data.y[idx])
        
        # Add final window
        if window_edges:
            window_data = Data(
                x=data.x,
                edge_index=torch.stack(window_edges, dim=1),
                edge_attr=torch.stack(window_attrs),
                edge_time=torch.tensor(window_times),
                y=torch.tensor(window_labels)
            )
            windows.append(window_data)
        
        return windows

class ConfigManager:
    """
    Manages configuration files and parameters for FORGE-AUTH.
    
    This class handles loading, validation, and merging of configuration files,
    ensuring consistent configuration across the authentication system.
    """
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from file with caching."""
        if config_name in self.config_cache:
            return self.config_cache[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        self._validate_config(config, config_name)
        
        # Apply defaults
        config = self._apply_defaults(config, config_name)
        
        # Cache configuration
        self.config_cache[config_name] = config
        
        return config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations with proper precedence."""
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _validate_config(self, config: Dict[str, Any], config_name: str):
        """Validate configuration structure and values."""
        required_fields = {
            'model': ['hidden_dim', 'num_layers'],
            'training': ['batch_size', 'learning_rate'],
            'privacy': ['use_differential_privacy', 'privacy_budget']
        }
        
        if config_name in required_fields:
            for field in required_fields[config_name]:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in {config_name} config")
    
    def _apply_defaults(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Apply default values to configuration."""
        defaults = {
            'model': {
                'dropout': 0.1,
                'activation': 'gelu',
                'use_batch_norm': True
            },
            'training': {
                'max_epochs': 100,
                'early_stopping_patience': 10,
                'gradient_clip': 1.0
            },
            'privacy': {
                'noise_multiplier': 1.0,
                'delta': 1e-5,
                'clipping_threshold': 1.0
            }
        }
        
        if config_name in defaults:
            return self._deep_merge(defaults[config_name], config)
        
        return config
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

class Visualizer:
    """
    Comprehensive visualization tools for FORGE-AUTH analysis.
    
    This class provides various visualization methods for analyzing authentication
    performance, privacy metrics, and system behavior using both static and
    interactive plots.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (10, 6)):
        self.style = style
        self.figsize = figsize
        
        # Set style
        plt.style.use(style)
        sns.set_palette(UTILS_CONSTANTS['COLOR_PALETTE'])
    
    def plot_authentication_metrics(self, metrics: Dict[str, List[float]], 
                                   save_path: Optional[str] = None):
        """Plot authentication performance metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metric_names):
            if metric in metrics:
                ax = axes[idx]
                values = metrics[metric]
                epochs = range(1, len(values) + 1)
                
                ax.plot(epochs, values, marker='o', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(epochs, values, 2)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "r--", alpha=0.5, label='Trend')
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=UTILS_CONSTANTS['FIGURE_DPI'], bbox_inches='tight')
            logger.info(f"Saved authentication metrics plot to {save_path}")
        
        plt.show()
    
    def plot_privacy_analysis(self, privacy_metrics: Dict[str, Any],
                            save_path: Optional[str] = None):
        """Visualize privacy analysis results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Privacy Budget Usage', 'Information Leakage',
                          'Attack Success Rates', 'Differential Privacy Guarantee'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Privacy budget usage over rounds
        if 'budget_usage' in privacy_metrics:
            rounds = list(range(len(privacy_metrics['budget_usage'])))
            fig.add_trace(
                go.Scatter(x=rounds, y=privacy_metrics['budget_usage'],
                          mode='lines+markers', name='Budget Used'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=rounds, y=[privacy_metrics['total_budget']] * len(rounds),
                          mode='lines', name='Budget Limit', line=dict(dash='dash')),
                row=1, col=1
            )
        
        # Information leakage
        if 'leakage_by_attribute' in privacy_metrics:
            attributes = list(privacy_metrics['leakage_by_attribute'].keys())
            leakage_values = list(privacy_metrics['leakage_by_attribute'].values())
            
            fig.add_trace(
                go.Bar(x=attributes, y=leakage_values, name='Leakage'),
                row=1, col=2
            )
        
        # Attack success rates
        if 'attack_results' in privacy_metrics:
            attacks = list(privacy_metrics['attack_results'].keys())
            success_rates = [v['success_rate'] for v in privacy_metrics['attack_results'].values()]
            
            fig.add_trace(
                go.Bar(x=attacks, y=success_rates, name='Success Rate'),
                row=2, col=1
            )
        
        # Differential privacy guarantee
        if 'epsilon_delta_curve' in privacy_metrics:
            epsilons = privacy_metrics['epsilon_delta_curve']['epsilon']
            deltas = privacy_metrics['epsilon_delta_curve']['delta']
            
            fig.add_trace(
                go.Scatter(x=epsilons, y=deltas, mode='lines+markers',
                          name='ε-δ Curve'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Privacy Analysis Dashboard")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved privacy analysis plot to {save_path}")
        
        fig.show()
    
    def plot_federated_learning_progress(self, fed_metrics: Dict[str, List[float]],
                                       save_path: Optional[str] = None):
        """Visualize federated learning progress."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Accuracy across rounds
        if 'round_accuracy' in fed_metrics:
            rounds = range(1, len(fed_metrics['round_accuracy']) + 1)
            ax1.plot(rounds, fed_metrics['round_accuracy'], 'b-', linewidth=2)
            ax1.fill_between(rounds, 
                           fed_metrics['round_accuracy'] - fed_metrics.get('accuracy_std', [0]*len(rounds)),
                           fed_metrics['round_accuracy'] + fed_metrics.get('accuracy_std', [0]*len(rounds)),
                           alpha=0.3)
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Global Model Accuracy')
            ax1.set_title('Federated Learning Convergence')
            ax1.grid(True, alpha=0.3)
        
        # Communication cost
        if 'communication_cost' in fed_metrics:
            rounds = range(1, len(fed_metrics['communication_cost']) + 1)
            ax2.bar(rounds, fed_metrics['communication_cost'], alpha=0.7)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Communication Cost (MB)')
            ax2.set_title('Communication Overhead per Round')
            ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=UTILS_CONSTANTS['FIGURE_DPI'])
            logger.info(f"Saved federated learning plot to {save_path}")
        
        plt.show()
    
    def plot_transaction_graph(self, graph_data: Data, highlight_fraud: bool = True,
                             save_path: Optional[str] = None):
        """Visualize transaction graph structure."""
        # Convert to NetworkX for visualization
        G = to_networkx(graph_data, to_undirected=True)
        
        plt.figure(figsize=(15, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        
        # Node colors based on features or fraud labels
        if highlight_fraud and hasattr(graph_data, 'y'):
            node_colors = ['red' if label == 1 else 'lightblue' 
                          for label in graph_data.y[:len(G.nodes())]]
        else:
            node_colors = 'lightblue'
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=50, alpha=0.7)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              alpha=0.5, width=0.5)
        
        plt.title("Transaction Graph Visualization")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=UTILS_CONSTANTS['FIGURE_DPI'], 
                       bbox_inches='tight')
            logger.info(f"Saved transaction graph to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, results: Dict[str, Any],
                                   save_path: str = "dashboard.html"):
        """Create interactive dashboard for experiment results."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Model Performance', 'Privacy Metrics',
                          'Attack Resistance', 'Resource Usage',
                          'Ablation Study', 'Comparison'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'table'}]]
        )
        
        # Add traces for each subplot based on available results
        if 'performance_history' in results:
            self._add_performance_traces(fig, results['performance_history'], row=1, col=1)
        
        if 'privacy_metrics' in results:
            self._add_privacy_traces(fig, results['privacy_metrics'], row=1, col=2)
        
        if 'attack_results' in results:
            self._add_attack_traces(fig, results['attack_results'], row=2, col=1)
        
        if 'resource_usage' in results:
            self._add_resource_traces(fig, results['resource_usage'], row=2, col=2)
        
        if 'ablation_results' in results:
            self._add_ablation_traces(fig, results['ablation_results'], row=3, col=1)
        
        if 'comparison_table' in results:
            self._add_comparison_table(fig, results['comparison_table'], row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="FORGE-AUTH Experiment Dashboard"
        )
        
        # Save dashboard
        fig.write_html(save_path)
        logger.info(f"Created interactive dashboard at {save_path}")
    
    def _add_performance_traces(self, fig, performance_data, row, col):
        """Add performance traces to dashboard."""
        epochs = list(range(len(performance_data['accuracy'])))
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in performance_data:
                fig.add_trace(
                    go.Scatter(x=epochs, y=performance_data[metric],
                             mode='lines+markers', name=metric.title()),
                    row=row, col=col
                )

class StatisticalAnalyzer:
    """
    Statistical analysis tools for experiment validation.
    
    This class provides rigorous statistical testing methods to ensure
    experimental results are significant and conclusions are valid.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_models(self, results1: Dict[str, List[float]], 
                      results2: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compare two models using statistical tests."""
        comparison_results = {}
        
        for metric in results1.keys():
            if metric in results2:
                values1 = np.array(results1[metric])
                values2 = np.array(results2[metric])
                
                # Normality test
                _, p_normal1 = stats.normaltest(values1)
                _, p_normal2 = stats.normaltest(values2)
                
                if p_normal1 > 0.05 and p_normal2 > 0.05:
                    # Use parametric test
                    statistic, p_value = stats.ttest_ind(values1, values2)
                    test_name = "t-test"
                else:
                    # Use non-parametric test
                    statistic, p_value = stats.mannwhitneyu(values1, values2)
                    test_name = "Mann-Whitney U"
                
                # Effect size
                effect_size = self._calculate_effect_size(values1, values2)
                
                comparison_results[metric] = {
                    'test': test_name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level,
                    'effect_size': effect_size,
                    'mean_difference': np.mean(values1) - np.mean(values2)
                }
        
        return comparison_results
    
    def _calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'bonferroni') -> np.ndarray:
        """Apply multiple comparison correction."""
        reject, p_adjusted, _, _ = multipletests(
            p_values, 
            alpha=self.significance_level, 
            method=method
        )
        
        return p_adjusted

class SystemProfiler:
    """
    System profiling utilities for performance optimization.
    
    This class provides tools for profiling code execution, identifying
    bottlenecks, and optimizing system performance.
    """
    
    def __init__(self):
        self.profile_data = {}
        
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code blocks."""
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        
        try:
            profiler.enable()
            yield
        finally:
            profiler.disable()
            
            # Get profile statistics
            stream = StringIO()
            ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            self.profile_data[name] = {
                'stats': stream.getvalue(),
                'total_time': ps.total_tt
            }
            
            logger.info(f"Profiled {name}: {ps.total_tt:.4f} seconds")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'hostname': socket.gethostname(),
            'username': getpass.getuser()
        }
        
        # GPU information
        if torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_devices'] = []
            
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                    'compute_capability': torch.cuda.get_device_capability(i)
                }
                info['gpu_devices'].append(gpu_info)
        
        return info

def create_reproducible_splits(dataset_size: int, train_ratio: float = 0.7,
                             val_ratio: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible train/validation/test splits."""
    np.random.seed(seed)
    indices = np.random.permutation(dataset_size)
    
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

def compute_authentication_metrics(predictions: np.ndarray, labels: np.ndarray,
                                 scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive authentication metrics."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, roc_auc_score, average_precision_score,
                                matthews_corrcoef)
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall': recall_score(labels, predictions, average='binary'),
        'f1_score': f1_score(labels, predictions, average='binary'),
        'mcc': matthews_corrcoef(labels, predictions)
    }
    
    if scores is not None:
        metrics['auc_roc'] = roc_auc_score(labels, scores)
        metrics['auc_pr'] = average_precision_score(labels, scores)
    
    # Authentication-specific metrics
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tp = np.sum((predictions == 1) & (labels == 1))
    
    metrics['false_accept_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_reject_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['equal_error_rate'] = (metrics['false_accept_rate'] + metrics['false_reject_rate']) / 2
    
    return metrics

def save_experiment_artifacts(artifacts: Dict[str, Any], experiment_id: str,
                            base_dir: str = "./experiments/artifacts"):
    """Save experiment artifacts with proper organization."""
    artifact_dir = Path(base_dir) / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    for name, artifact in artifacts.items():
        if isinstance(artifact, torch.nn.Module):
            # Save model
            torch.save(artifact.state_dict(), artifact_dir / f"{name}.pt")
        elif isinstance(artifact, pd.DataFrame):
            # Save dataframe
            artifact.to_csv(artifact_dir / f"{name}.csv", index=False)
        elif isinstance(artifact, dict):
            # Save dictionary as JSON
            with open(artifact_dir / f"{name}.json", 'w') as f:
                json.dump(artifact, f, indent=2, default=str)
        elif isinstance(artifact, np.ndarray):
            # Save numpy array
            np.save(artifact_dir / f"{name}.npy", artifact)
        else:
            # Save as pickle for other types
            with open(artifact_dir / f"{name}.pkl", 'wb') as f:
                pickle.dump(artifact, f)
    
    logger.info(f"Saved {len(artifacts)} artifacts to {artifact_dir}")

# Export main utilities
__all__ = [
    'set_random_seeds',
    'create_logger',
    'Timer',
    'MemoryTracker',
    'DataGenerator',
    'ConfigManager',
    'Visualizer',
    'StatisticalAnalyzer',
    'SystemProfiler',
    'create_reproducible_splits',
    'compute_authentication_metrics',
    'save_experiment_artifacts',
    'UTILS_CONSTANTS'
]