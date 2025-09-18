"""
FORGE-AUTH: Complete Experimental Pipeline
==========================================

This module implements the comprehensive experimental framework for evaluating
FORGE-AUTH's privacy-preserving authentication system. It provides rigorous
benchmarking, privacy analysis, attack simulations, and performance evaluation
across various scenarios including centralized, federated, and adversarial settings.

Key Components:
- Experiment configuration and management
- Dataset generation and preprocessing pipelines
- Training and evaluation frameworks
- Privacy measurement and verification
- Attack simulation and robustness testing
- Performance profiling and optimization
- Statistical analysis and visualization
- Reproducibility mechanisms

Author: FORGE-AUTH Research Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch, TemporalData
from torch_geometric.loader import TemporalDataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict, deque
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time
import logging
import json
import pickle
import os
import shutil
import hashlib
import random
import warnings
from pathlib import Path
from datetime import datetime
import wandb
from tqdm import tqdm
import yaml
import argparse
from abc import ABC, abstractmethod
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import psutil
import traceback

# Scientific computing
from scipy import stats
from scipy.optimize import differential_evolution
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Import FORGE-AUTH modules
from forge_auth_core import (
    DistributedAuthenticationEngine,
    AuthenticationRequest,
    AuthenticationMode,
    CredentialType,
    create_authentication_engine
)
from forge_auth_models import (
    create_authentication_model,
    compute_model_privacy_guarantees,
    ModelCheckpointer,
    ModelMode
)
from forge_auth_federated import (
    create_federated_orchestrator,
    FederatedConfig,
    AggregationStrategy,
    ClientSelectionStrategy
)
from forge_auth_utils import (
    set_random_seeds,
    create_logger,
    Timer,
    MemoryTracker,
    DataGenerator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experiment constants
EXPERIMENT_CONSTANTS = {
    'RANDOM_SEED': 42,
    'NUM_WORKERS': 4,
    'CHECKPOINT_DIR': './experiments/checkpoints',
    'RESULTS_DIR': './experiments/results',
    'DATA_DIR': './experiments/data',
    'LOG_DIR': './experiments/logs',
    'TENSORBOARD_DIR': './experiments/tensorboard',
    'MAX_EPOCHS': 100,
    'EARLY_STOPPING_PATIENCE': 10,
    'EVALUATION_FREQUENCY': 5,
    'PRIVACY_EVALUATION_ROUNDS': 100,
    'ATTACK_ITERATIONS': 1000,
    'STATISTICAL_SIGNIFICANCE': 0.05
}

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Experiment metadata
    experiment_name: str
    experiment_type: str  # 'centralized', 'federated', 'privacy', 'attack', 'ablation'
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Dataset configuration
    dataset_name: str = "forge_auth_synthetic"
    num_users: int = 10000
    num_transactions: int = 1000000
    temporal_window: int = 86400  # 24 hours
    fraud_rate: float = 0.01
    
    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adam"
    scheduler: str = "cosine"
    max_epochs: int = EXPERIMENT_CONSTANTS['MAX_EPOCHS']
    gradient_clip: float = 1.0
    
    # Privacy configuration
    use_differential_privacy: bool = True
    privacy_budget: float = 10.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    
    # Federated configuration
    federated_config: Optional[Dict[str, Any]] = None
    
    # Evaluation configuration
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'auc_roc', 'precision', 'recall', 'f1',
        'false_positive_rate', 'false_negative_rate'
    ])
    
    # Hardware configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count()
    mixed_precision: bool = True
    
    # Reproducibility
    random_seed: int = EXPERIMENT_CONSTANTS['RANDOM_SEED']
    deterministic: bool = True
    
    # Logging configuration
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "forge-auth"
    use_tensorboard: bool = True
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    keep_best_only: bool = True
    
    # Resource limits
    max_memory_gb: float = 32.0
    timeout_hours: float = 24.0

class ExperimentRunner:
    """Main experiment runner for FORGE-AUTH"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.start_time = time.time()
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize tracking
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.experiment_id = self._generate_experiment_id()
        
        # Setup device
        self.device = torch.device(config.device)
        if config.device == "cuda":
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        
        # Initialize experiment tracking
        if config.use_wandb:
            self._init_wandb()
        if config.use_tensorboard:
            self.writer = SummaryWriter(
                os.path.join(EXPERIMENT_CONSTANTS['TENSORBOARD_DIR'], 
                           self.experiment_id)
            )
        
        # Memory tracking
        self.memory_tracker = MemoryTracker(max_memory_gb=config.max_memory_gb)
        
    def _setup_directories(self):
        """Create experiment directories"""
        for dir_path in [
            EXPERIMENT_CONSTANTS['CHECKPOINT_DIR'],
            EXPERIMENT_CONSTANTS['RESULTS_DIR'],
            EXPERIMENT_CONSTANTS['DATA_DIR'],
            EXPERIMENT_CONSTANTS['LOG_DIR'],
            EXPERIMENT_CONSTANTS['TENSORBOARD_DIR']
        ]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup experiment logging"""
        log_file = os.path.join(
            EXPERIMENT_CONSTANTS['LOG_DIR'],
            f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        return create_logger(
            name=self.config.experiment_name,
            log_file=log_file,
            level=logging.INFO
        )
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        set_random_seeds(self.config.random_seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = hashlib.md5(
            json.dumps(self.config.__dict__, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"{self.config.experiment_name}_{timestamp}_{config_hash}"
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project=self.config.wandb_project,
            name=self.experiment_id,
            config=self.config.__dict__,
            tags=self.config.tags
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment"""
        self.logger.info(f"Starting experiment: {self.experiment_id}")
        self.logger.info(f"Configuration: {json.dumps(self.config.__dict__, indent=2)}")
        
        try:
            # Run experiment based on type
            if self.config.experiment_type == "centralized":
                results = self._run_centralized_experiment()
            elif self.config.experiment_type == "federated":
                results = self._run_federated_experiment()
            elif self.config.experiment_type == "privacy":
                results = self._run_privacy_experiment()
            elif self.config.experiment_type == "attack":
                results = self._run_attack_experiment()
            elif self.config.experiment_type == "ablation":
                results = self._run_ablation_study()
            else:
                raise ValueError(f"Unknown experiment type: {self.config.experiment_type}")
            
            # Add metadata
            results['experiment_id'] = self.experiment_id
            results['config'] = self.config.__dict__
            results['duration'] = time.time() - self.start_time
            
            # Save results
            self._save_results(results)
            
            # Cleanup
            self._cleanup()
            
            self.logger.info(f"Experiment completed successfully in {results['duration']:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _run_centralized_experiment(self) -> Dict[str, Any]:
        """Run centralized authentication experiment"""
        self.logger.info("Running centralized authentication experiment")
        
        # Create dataset
        train_dataset, val_dataset, test_dataset = self._create_datasets()
        
        # Create data loaders
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)
        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        
        # Create model
        model = create_authentication_model(self.config.model_config)
        model = model.to(self.device)
        
        # Create optimizer and scheduler
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        
        # Create authentication engine
        auth_engine = create_authentication_engine(
            graph_engine=None,  # Will be initialized with data
            config=self.config.model_config
        )
        
        # Training loop
        best_val_metric = 0
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, epoch
            )
            
            # Validate
            val_metrics = self._evaluate(
                model, val_loader, auth_engine, epoch
            )
            
            # Update scheduler
            if scheduler:
                scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Early stopping
            if val_metrics['auc_roc'] > best_val_metric:
                best_val_metric = val_metrics['auc_roc']
                patience_counter = 0
                
                # Save best model
                if self.config.save_checkpoints:
                    self._save_checkpoint(model, optimizer, epoch, val_metrics)
            else:
                patience_counter += 1
                
            if patience_counter >= EXPERIMENT_CONSTANTS['EARLY_STOPPING_PATIENCE']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Test evaluation
        test_metrics = self._evaluate(
            model, test_loader, auth_engine, epoch, is_test=True
        )
        
        # Privacy analysis
        privacy_metrics = self._analyze_privacy(model, test_loader)
        
        return {
            'best_val_metrics': self.best_metrics,
            'test_metrics': test_metrics,
            'privacy_metrics': privacy_metrics,
            'training_history': dict(self.metrics_history),
            'model_params': sum(p.numel() for p in model.parameters())
        }
    
    def _run_federated_experiment(self) -> Dict[str, Any]:
        """Run federated learning experiment"""
        self.logger.info("Running federated learning experiment")
        
        # Create federated configuration
        fed_config = FederatedConfig(**self.config.federated_config)
        
        # Create global model
        global_model = create_authentication_model(self.config.model_config)
        global_model = global_model.to(self.device)
        
        # Create federated orchestrator
        orchestrator = create_federated_orchestrator(
            self.config.federated_config,
            global_model
        )
        
        # Simulate clients
        client_datasets = self._create_federated_datasets(
            num_clients=fed_config.num_rounds
        )
        
        # Register clients
        for client_id, client_data in client_datasets.items():
            orchestrator.register_client(
                client_id,
                {'num_samples': len(client_data)}
            )
        
        # Federated training
        federated_metrics = defaultdict(list)
        
        for round_num in range(fed_config.num_rounds):
            self.logger.info(f"Federated round {round_num + 1}/{fed_config.num_rounds}")
            
            # Run federated round
            round_info = orchestrator.run_federated_round()
            
            # Evaluate global model
            if round_num % EXPERIMENT_CONSTANTS['EVALUATION_FREQUENCY'] == 0:
                eval_metrics = self._evaluate_federated_model(
                    global_model,
                    client_datasets,
                    round_num
                )
                
                for metric_name, value in eval_metrics.items():
                    federated_metrics[metric_name].append(value)
            
            # Log round metrics
            self._log_federated_round(round_info, round_num)
        
        # Final evaluation
        final_metrics = self._evaluate_federated_model(
            global_model,
            client_datasets,
            fed_config.num_rounds,
            is_final=True
        )
        
        # Privacy analysis
        privacy_guarantee = orchestrator.get_privacy_guarantee()
        communication_stats = orchestrator.get_communication_stats()
        
        return {
            'federated_metrics': dict(federated_metrics),
            'final_metrics': final_metrics,
            'privacy_guarantee': privacy_guarantee,
            'communication_stats': communication_stats,
            'num_rounds': fed_config.num_rounds
        }
    
    def _run_privacy_experiment(self) -> Dict[str, Any]:
        """Run privacy analysis experiment"""
        self.logger.info("Running privacy analysis experiment")
        
        # Create dataset
        _, _, test_dataset = self._create_datasets()
        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        
        # Create model
        model = create_authentication_model(self.config.model_config)
        model = model.to(self.device)
        
        # Privacy attacks
        privacy_results = {}
        
        # Membership inference attack
        self.logger.info("Running membership inference attack")
        mia_results = self._membership_inference_attack(
            model, test_loader
        )
        privacy_results['membership_inference'] = mia_results
        
        # Model inversion attack
        self.logger.info("Running model inversion attack")
        inversion_results = self._model_inversion_attack(
            model, test_loader
        )
        privacy_results['model_inversion'] = inversion_results
        
        # Attribute inference attack
        self.logger.info("Running attribute inference attack")
        attribute_results = self._attribute_inference_attack(
            model, test_loader
        )
        privacy_results['attribute_inference'] = attribute_results
        
        # Differential privacy analysis
        self.logger.info("Analyzing differential privacy guarantees")
        dp_analysis = self._differential_privacy_analysis(
            model, test_loader
        )
        privacy_results['differential_privacy'] = dp_analysis
        
        # Information leakage measurement
        self.logger.info("Measuring information leakage")
        leakage_results = self._measure_information_leakage(
            model, test_loader
        )
        privacy_results['information_leakage'] = leakage_results
        
        return privacy_results
    
    def _run_attack_experiment(self) -> Dict[str, Any]:
        """Run adversarial attack experiment"""
        self.logger.info("Running adversarial attack experiment")
        
        # Create dataset
        _, _, test_dataset = self._create_datasets()
        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        
        # Create model
        model = create_authentication_model(self.config.model_config)
        model = model.to(self.device)
        
        # Load pretrained weights if available
        checkpoint_path = self._get_best_checkpoint()
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded pretrained model from {checkpoint_path}")
        
        attack_results = {}
        
        # Evasion attacks
        self.logger.info("Running evasion attacks")
        evasion_results = self._run_evasion_attacks(model, test_loader)
        attack_results['evasion'] = evasion_results
        
        # Poisoning attacks
        self.logger.info("Running poisoning attacks")
        poisoning_results = self._run_poisoning_attacks(model, test_loader)
        attack_results['poisoning'] = poisoning_results
        
        # Byzantine attacks (for federated setting)
        self.logger.info("Running Byzantine attacks")
        byzantine_results = self._run_byzantine_attacks(model)
        attack_results['byzantine'] = byzantine_results
        
        # Sybil attacks
        self.logger.info("Running Sybil attacks")
        sybil_results = self._run_sybil_attacks(model)
        attack_results['sybil'] = sybil_results
        
        # Defense evaluation
        self.logger.info("Evaluating defenses")
        defense_results = self._evaluate_defenses(model, attack_results)
        attack_results['defenses'] = defense_results
        
        return attack_results
    
    def _run_ablation_study(self) -> Dict[str, Any]:
        """Run ablation study"""
        self.logger.info("Running ablation study")
        
        # Define components to ablate
        ablation_components = [
            'temporal_attention',
            'behavioral_biometrics',
            'selective_disclosure',
            'differential_privacy',
            'secure_aggregation',
            'byzantine_robustness'
        ]
        
        ablation_results = {}
        
        # Baseline (all components)
        self.logger.info("Training baseline model with all components")
        baseline_config = self.config.model_config.copy()
        baseline_results = self._train_and_evaluate_model(baseline_config)
        ablation_results['baseline'] = baseline_results
        
        # Ablate each component
        for component in ablation_components:
            self.logger.info(f"Ablating component: {component}")
            
            # Create configuration without component
            ablated_config = self._create_ablated_config(
                baseline_config, component
            )
            
            # Train and evaluate
            component_results = self._train_and_evaluate_model(ablated_config)
            ablation_results[f'without_{component}'] = component_results
            
            # Calculate impact
            impact = self._calculate_ablation_impact(
                baseline_results, component_results
            )
            ablation_results[f'{component}_impact'] = impact
        
        # Statistical analysis
        statistical_analysis = self._ablation_statistical_analysis(
            ablation_results
        )
        ablation_results['statistical_analysis'] = statistical_analysis
        
        # Visualization
        self._visualize_ablation_results(ablation_results)
        
        return ablation_results
    
    def _create_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train, validation, and test datasets"""
        self.logger.info("Creating datasets")
        
        # Generate synthetic data
        data_generator = DataGenerator(
            num_users=self.config.num_users,
            num_transactions=self.config.num_transactions,
            fraud_rate=self.config.fraud_rate
        )
        
        # Generate temporal graph data
        graph_data = data_generator.generate_temporal_graph()
        
        # Split data
        train_ratio, val_ratio = 0.7, 0.15
        num_samples = len(graph_data)
        
        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        
        indices = torch.randperm(num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = TemporalGraphDataset(graph_data, train_indices)
        val_dataset = TemporalGraphDataset(graph_data, val_indices)
        test_dataset = TemporalGraphDataset(graph_data, test_indices)
        
        self.logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
                        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_federated_datasets(self, num_clients: int) -> Dict[str, Dataset]:
        """Create datasets for federated clients"""
        self.logger.info(f"Creating federated datasets for {num_clients} clients")
        
        # Generate data with non-IID distribution
        client_datasets = {}
        
        for client_id in range(num_clients):
            # Vary data characteristics per client
            client_fraud_rate = np.random.beta(2, 5)  # Skewed distribution
            client_transactions = np.random.randint(1000, 10000)
            
            data_generator = DataGenerator(
                num_users=self.config.num_users // num_clients,
                num_transactions=client_transactions,
                fraud_rate=client_fraud_rate
            )
            
            client_data = data_generator.generate_temporal_graph()
            client_datasets[f"client_{client_id}"] = TemporalGraphDataset(
                client_data,
                torch.arange(len(client_data))
            )
        
        return client_datasets
    
    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create data loader"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=EXPERIMENT_CONSTANTS['NUM_WORKERS'],
            pin_memory=True if self.config.device == "cuda" else False,
            drop_last=False
        )
    
    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer"""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler"""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        else:
            return None
    
    def _train_epoch(self, model: nn.Module, dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.gradient_clip
                )
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = outputs['predictions']
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct / total:.4f}"
            })
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                self.memory_tracker.check_memory()
                gc.collect()
                torch.cuda.empty_cache()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def _evaluate(self, model: nn.Module, dataloader: DataLoader,
                 auth_engine: Any, epoch: int, is_test: bool = False) -> Dict[str, float]:
        """Evaluate model"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_scores = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = self._batch_to_device(batch)
                
                # Forward pass
                outputs = model(batch)
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                
                # Collect predictions
                predictions = outputs['predictions'].cpu().numpy()
                scores = outputs['probabilities'][:, 1].cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_scores.extend(scores)
                all_labels.extend(labels)
        
        # Compute metrics
        metrics = self._compute_metrics(
            np.array(all_predictions),
            np.array(all_scores),
            np.array(all_labels)
        )
        
        metrics['loss'] = total_loss / len(dataloader)
        
        # Authentication-specific evaluation
        if is_test:
            auth_metrics = self._evaluate_authentication(
                model, dataloader, auth_engine
            )
            metrics.update(auth_metrics)
        
        # Update best metrics
        if not is_test:
            for metric_name, value in metrics.items():
                if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
        
        return metrics
    
    def _evaluate_authentication(self, model: nn.Module, dataloader: DataLoader,
                                auth_engine: Any) -> Dict[str, float]:
        """Evaluate authentication performance"""
        auth_metrics = {
            'authentication_latency': [],
            'privacy_cost': [],
            'false_acceptance_rate': 0,
            'false_rejection_rate': 0
        }
        
        num_requests = 100
        
        for i in range(num_requests):
            # Create authentication request
            auth_request = self._create_auth_request()
            
            # Measure authentication time
            start_time = time.time()
            authenticated, proof = auth_engine.authenticate(auth_request)
            latency = time.time() - start_time
            
            auth_metrics['authentication_latency'].append(latency)
            auth_metrics['privacy_cost'].append(proof.privacy_cost)
        
        # Compute statistics
        auth_metrics['mean_latency'] = np.mean(auth_metrics['authentication_latency'])
        auth_metrics['p95_latency'] = np.percentile(auth_metrics['authentication_latency'], 95)
        auth_metrics['mean_privacy_cost'] = np.mean(auth_metrics['privacy_cost'])
        
        return auth_metrics
    
    def _membership_inference_attack(self, model: nn.Module,
                                   dataloader: DataLoader) -> Dict[str, float]:
        """Perform membership inference attack"""
        # Create shadow models
        num_shadow_models = 5
        shadow_models = []
        
        for i in range(num_shadow_models):
            shadow_model = create_authentication_model(self.config.model_config)
            shadow_model = shadow_model.to(self.device)
            
            # Train shadow model on subset of data
            self._train_shadow_model(shadow_model, dataloader)
            shadow_models.append(shadow_model)
        
        # Attack model training
        attack_dataset = self._create_attack_dataset(
            model, shadow_models, dataloader
        )
        
        # Train attack model
        attack_model = self._train_attack_model(attack_dataset)
        
        # Evaluate attack
        attack_accuracy = self._evaluate_attack_model(
            attack_model, model, dataloader
        )
        
        return {
            'attack_accuracy': attack_accuracy,
            'privacy_risk': attack_accuracy - 0.5,  # Above random guessing
            'num_shadow_models': num_shadow_models
        }
    
    def _model_inversion_attack(self, model: nn.Module,
                              dataloader: DataLoader) -> Dict[str, float]:
        """Perform model inversion attack"""
        # Gradient-based inversion
        inversion_results = []
        
        for class_label in [0, 1]:  # Binary classification
            # Initialize with random input
            inverted_input = torch.randn(1, self.config.model_config['credential_dim'])
            inverted_input = inverted_input.to(self.device)
            inverted_input.requires_grad = True
            
            # Optimization loop
            optimizer = torch.optim.LBFGS([inverted_input])
            
            def closure():
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model({'features': inverted_input})
                
                # Target specific class
                target = torch.tensor([class_label]).to(self.device)
                loss = F.cross_entropy(outputs['logits'], target)
                
                loss.backward()
                return loss
            
            for _ in range(100):
                optimizer.step(closure)
            
            # Measure reconstruction quality
            reconstruction_error = self._measure_reconstruction_error(
                inverted_input, dataloader, class_label
            )
            
            inversion_results.append(reconstruction_error)
        
        return {
            'mean_reconstruction_error': np.mean(inversion_results),
            'max_reconstruction_error': np.max(inversion_results)
        }
    
    def _differential_privacy_analysis(self, model: nn.Module,
                                     dataloader: DataLoader) -> Dict[str, float]:
        """Analyze differential privacy guarantees"""
        # Compute privacy guarantees
        privacy_params = compute_model_privacy_guarantees(
            model=model,
            dataset_size=len(dataloader.dataset),
            batch_size=self.config.batch_size,
            epochs=self.config.max_epochs,
            noise_multiplier=self.config.noise_multiplier
        )
        
        # Empirical privacy analysis
        empirical_epsilon = self._compute_empirical_epsilon(
            model, dataloader
        )
        
        return {
            'theoretical_epsilon': privacy_params['epsilon'],
            'theoretical_delta': privacy_params['delta'],
            'empirical_epsilon': empirical_epsilon,
            'privacy_budget_used': privacy_params['epsilon'],
            'privacy_budget_remaining': max(0, self.config.privacy_budget - privacy_params['epsilon'])
        }
    
    def _measure_information_leakage(self, model: nn.Module,
                                   dataloader: DataLoader) -> Dict[str, float]:
        """Measure information leakage from model"""
        # Mutual information estimation
        mutual_info = self._estimate_mutual_information(model, dataloader)
        
        # Channel capacity
        channel_capacity = self._compute_channel_capacity(model, dataloader)
        
        # Attribute disclosure risk
        disclosure_risk = self._compute_disclosure_risk(model, dataloader)
        
        return {
            'mutual_information': mutual_info,
            'channel_capacity': channel_capacity,
            'attribute_disclosure_risk': disclosure_risk,
            'information_leakage_score': (mutual_info + disclosure_risk) / 2
        }
    
    def _run_evasion_attacks(self, model: nn.Module,
                           dataloader: DataLoader) -> Dict[str, Any]:
        """Run evasion attacks against model"""
        evasion_methods = ['fgsm', 'pgd', 'cw', 'deepfool']
        results = {}
        
        for method in evasion_methods:
            self.logger.info(f"Running {method} attack")
            
            if method == 'fgsm':
                attack_results = self._fgsm_attack(model, dataloader)
            elif method == 'pgd':
                attack_results = self._pgd_attack(model, dataloader)
            elif method == 'cw':
                attack_results = self._cw_attack(model, dataloader)
            elif method == 'deepfool':
                attack_results = self._deepfool_attack(model, dataloader)
            
            results[method] = attack_results
        
        return results
    
    def _fgsm_attack(self, model: nn.Module, dataloader: DataLoader,
                    epsilon: float = 0.1) -> Dict[str, float]:
        """Fast Gradient Sign Method attack"""
        model.eval()
        
        correct_before = 0
        correct_after = 0
        total = 0
        
        for batch in dataloader:
            batch = self._batch_to_device(batch)
            batch['features'].requires_grad = True
            
            # Original prediction
            outputs = model(batch)
            original_pred = outputs['predictions']
            
            # Compute loss
            loss = F.cross_entropy(outputs['logits'], batch['labels'])
            model.zero_grad()
            loss.backward()
            
            # Generate adversarial examples
            data_grad = batch['features'].grad.data
            perturbed_data = batch['features'] + epsilon * data_grad.sign()
            
            # Adversarial prediction
            batch['features'] = perturbed_data
            adv_outputs = model(batch)
            adv_pred = adv_outputs['predictions']
            
            # Update statistics
            correct_before += (original_pred == batch['labels']).sum().item()
            correct_after += (adv_pred == batch['labels']).sum().item()
            total += batch['labels'].size(0)
        
        return {
            'accuracy_before': correct_before / total,
            'accuracy_after': correct_after / total,
            'attack_success_rate': 1 - (correct_after / total),
            'epsilon': epsilon
        }
    
    def _visualize_ablation_results(self, results: Dict[str, Any]):
        """Visualize ablation study results"""
        # Extract metrics
        components = []
        impacts = []
        
        for key, value in results.items():
            if '_impact' in key:
                component = key.replace('_impact', '')
                components.append(component)
                impacts.append(value['accuracy_impact'])
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(components, impacts)
        
        # Color bars based on impact
        colors = ['red' if impact < 0 else 'green' for impact in impacts]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Component')
        plt.ylabel('Impact on Accuracy')
        plt.title('Ablation Study: Component Impact Analysis')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            EXPERIMENT_CONSTANTS['RESULTS_DIR'],
            f"{self.experiment_id}_ablation.png"
        )
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved ablation plot to {plot_path}")
    
    def _compute_metrics(self, predictions: np.ndarray, scores: np.ndarray,
                        labels: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(predictions == labels)
        
        # For binary classification
        if len(np.unique(labels)) == 2:
            metrics['auc_roc'] = roc_auc_score(labels, scores)
            
            # Compute optimal threshold
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Metrics at optimal threshold
            optimal_predictions = (scores > optimal_threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(labels, optimal_predictions).ravel()
            
            metrics['precision'] = tp / (tp + fp + 1e-10)
            metrics['recall'] = tp / (tp + fn + 1e-10)
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
                metrics['precision'] + metrics['recall'] + 1e-10
            )
            metrics['false_positive_rate'] = fp / (fp + tn + 1e-10)
            metrics['false_negative_rate'] = fn / (fn + tp + 1e-10)
            
        return metrics
    
    def _save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(
            EXPERIMENT_CONSTANTS['CHECKPOINT_DIR'],
            f"{self.experiment_id}_epoch_{epoch}.pt"
        )
        
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only best checkpoint if configured
        if self.config.keep_best_only:
            # Remove previous checkpoints
            for old_checkpoint in os.listdir(EXPERIMENT_CONSTANTS['CHECKPOINT_DIR']):
                if self.experiment_id in old_checkpoint and old_checkpoint != os.path.basename(checkpoint_path):
                    os.remove(os.path.join(EXPERIMENT_CONSTANTS['CHECKPOINT_DIR'], old_checkpoint))
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results"""
        # Save as JSON
        json_path = os.path.join(
            EXPERIMENT_CONSTANTS['RESULTS_DIR'],
            f"{self.experiment_id}_results.json"
        )
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save as pickle for complex objects
        pickle_path = os.path.join(
            EXPERIMENT_CONSTANTS['RESULTS_DIR'],
            f"{self.experiment_id}_results.pkl"
        )
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Saved results to {json_path} and {pickle_path}")
    
    def _cleanup(self):
        """Cleanup resources"""
        # Close tensorboard writer
        if hasattr(self, 'writer'):
            self.writer.close()
        
        # Close wandb
        if self.config.use_wandb:
            wandb.finish()
        
        # Clear GPU cache
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        
        # Garbage collection
        gc.collect()

class TemporalGraphDataset(Dataset):
    """Dataset for temporal graph data"""
    
    def __init__(self, graph_data: List[Data], indices: torch.Tensor):
        self.graph_data = graph_data
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.graph_data[actual_idx]

class ExperimentOrchestrator:
    """Orchestrates multiple experiments"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.experiments = self._load_experiments()
        self.results = {}
        
    def _load_experiments(self) -> List[ExperimentConfig]:
        """Load experiment configurations from file"""
        with open(self.config_file, 'r') as f:
            configs = yaml.safe_load(f)
        
        experiments = []
        for exp_config in configs['experiments']:
            experiments.append(ExperimentConfig(**exp_config))
        
        return experiments
    
    def run_all(self, parallel: bool = False):
        """Run all experiments"""
        if parallel:
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for exp_config in self.experiments:
                    future = executor.submit(self._run_single_experiment, exp_config)
                    futures.append((exp_config.experiment_name, future))
                
                for exp_name, future in futures:
                    try:
                        result = future.result()
                        self.results[exp_name] = result
                    except Exception as e:
                        logger.error(f"Experiment {exp_name} failed: {e}")
        else:
            for exp_config in self.experiments:
                try:
                    result = self._run_single_experiment(exp_config)
                    self.results[exp_config.experiment_name] = result
                except Exception as e:
                    logger.error(f"Experiment {exp_config.experiment_name} failed: {e}")
    
    def _run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run single experiment"""
        runner = ExperimentRunner(config)
        return runner.run()
    
    def generate_report(self):
        """Generate comprehensive experiment report"""
        report = {
            'summary': self._generate_summary(),
            'detailed_results': self.results,
            'comparisons': self._generate_comparisons(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = os.path.join(
            EXPERIMENT_CONSTANTS['RESULTS_DIR'],
            f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated report at {report_path}")
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all experiments"""
        summary = {
            'total_experiments': len(self.experiments),
            'completed_experiments': len(self.results),
            'failed_experiments': len(self.experiments) - len(self.results),
            'best_results': {},
            'aggregate_metrics': {}
        }
        
        # Find best results
        for metric in ['accuracy', 'auc_roc', 'f1']:
            best_value = 0
            best_experiment = None
            
            for exp_name, results in self.results.items():
                if 'test_metrics' in results and metric in results['test_metrics']:
                    value = results['test_metrics'][metric]
                    if value > best_value:
                        best_value = value
                        best_experiment = exp_name
            
            summary['best_results'][metric] = {
                'experiment': best_experiment,
                'value': best_value
            }
        
        return summary
    
    def _generate_comparisons(self) -> Dict[str, Any]:
        """Generate comparisons between experiments"""
        comparisons = {}
        
        # Compare centralized vs federated
        if 'centralized' in self.results and 'federated' in self.results:
            comparisons['centralized_vs_federated'] = {
                'accuracy_difference': (
                    self.results['centralized']['test_metrics']['accuracy'] -
                    self.results['federated']['final_metrics']['accuracy']
                ),
                'privacy_improvement': (
                    self.results['federated']['privacy_guarantee']['epsilon'] /
                    self.results['centralized']['privacy_metrics']['differential_privacy']['theoretical_epsilon']
                )
            }
        
        return comparisons
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Check privacy vs accuracy tradeoff
        for exp_name, results in self.results.items():
            if 'privacy_metrics' in results:
                epsilon = results['privacy_metrics']['differential_privacy']['theoretical_epsilon']
                accuracy = results['test_metrics']['accuracy']
                
                if epsilon > 10 and accuracy < 0.9:
                    recommendations.append(
                        f"Experiment {exp_name}: Consider reducing privacy budget "
                        f"for better accuracy (current ε={epsilon:.2f}, acc={accuracy:.3f})"
                    )
        
        return recommendations

def main():
    """Main entry point for experiments"""
    parser = argparse.ArgumentParser(description="FORGE-AUTH Experiments")
    parser.add_argument('--config', type=str, required=True,
                      help='Path to experiment configuration file')
    parser.add_argument('--parallel', action='store_true',
                      help='Run experiments in parallel')
    parser.add_argument('--experiment', type=str,
                      help='Run specific experiment only')
    
    args = parser.parse_args()
    
    if args.experiment:
        # Run single experiment
        with open(args.config, 'r') as f:
            configs = yaml.safe_load(f)
        
        exp_config = None
        for config in configs['experiments']:
            if config['experiment_name'] == args.experiment:
                exp_config = ExperimentConfig(**config)
                break
        
        if exp_config:
            runner = ExperimentRunner(exp_config)
            results = runner.run()
            print(f"Experiment completed. Results: {results}")
        else:
            print(f"Experiment {args.experiment} not found in configuration")
    else:
        # Run all experiments
        orchestrator = ExperimentOrchestrator(args.config)
        orchestrator.run_all(parallel=args.parallel)
        report = orchestrator.generate_report()
        print(f"All experiments completed. Report generated: {report}")

if __name__ == "__main__":
    main()

# Export main components
__all__ = [
    'ExperimentConfig',
    'ExperimentRunner',
    'ExperimentOrchestrator',
    'TemporalGraphDataset',
    'main'
]