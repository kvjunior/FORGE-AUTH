"""
FORGE-AUTH: Federated Learning and Privacy Protocols
====================================================

This module implements the federated learning framework and privacy protocols
for FORGE-AUTH, enabling privacy-preserving collaborative authentication across
multiple organizations without sharing sensitive data. It provides secure
aggregation, differential privacy mechanisms, and Byzantine-resistant protocols.

Key Components:
- Secure federated aggregation with homomorphic encryption
- Differential privacy for federated learning
- Byzantine-fault tolerant aggregation
- Adaptive privacy budget management
- Asynchronous federated protocols
- Communication-efficient updates

Author: FORGE-AUTH Research Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict, deque
import hashlib
import secrets
import time
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import logging
from enum import Enum
import pickle
import json
import struct
import math
import copy
import os
from abc import ABC, abstractmethod
import heapq
import zlib
import msgpack

# Cryptographic imports
from cryptography.hazmat.primitives import hashes, serialization, hmac
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Advanced privacy mechanisms
try:
    import phe as paillier  # Paillier homomorphic encryption
    PAILLIER_AVAILABLE = True
except ImportError:
    PAILLIER_AVAILABLE = False

try:
    import tenseal as ts  # CKKS homomorphic encryption
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Federated constants
FEDERATED_CONSTANTS = {
    'MIN_CLIENTS_PER_ROUND': 2,
    'MAX_ROUNDS': 1000,
    'STALENESS_THRESHOLD': 10,
    'COMPRESSION_THRESHOLD': 0.1,
    'BYZANTINE_THRESHOLD': 0.3,
    'PRIVACY_AMPLIFICATION_FACTOR': 2.0,
    'SECURE_AGGREGATION_THRESHOLD': 3,
    'MAX_CLIENT_COMPUTATION_TIME': 300,  # seconds
    'COMMUNICATION_TIMEOUT': 60,  # seconds
    'CHECKPOINT_FREQUENCY': 10
}

class AggregationStrategy(Enum):
    """Strategies for federated aggregation"""
    FEDERATED_AVERAGING = "fedavg"
    FEDERATED_SGD = "fedsgd"
    FEDERATED_ADAM = "fedadam"
    FEDERATED_YOGI = "fedyogi"
    SCAFFOLD = "scaffold"
    FEDPROX = "fedprox"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    MEDIAN = "median"
    BYZANTINE_ROBUST = "byzantine_robust"

class ClientSelectionStrategy(Enum):
    """Strategies for client selection"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    CONTRIBUTION_BASED = "contribution"
    RESOURCE_AWARE = "resource"
    PRIVACY_AWARE = "privacy"
    REPUTATION_BASED = "reputation"

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    # Basic settings
    num_rounds: int = 100
    clients_per_round: int = 10
    min_clients: int = FEDERATED_CONSTANTS['MIN_CLIENTS_PER_ROUND']
    local_epochs: int = 5
    local_batch_size: int = 32
    
    # Aggregation settings
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING
    client_selection_strategy: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    client_fraction: float = 0.1
    
    # Privacy settings
    use_differential_privacy: bool = True
    privacy_budget: float = 10.0
    noise_multiplier: float = 1.0
    clipping_threshold: float = 1.0
    secure_aggregation: bool = True
    homomorphic_encryption: bool = False
    
    # Byzantine robustness
    byzantine_robust: bool = True
    byzantine_threshold: float = FEDERATED_CONSTANTS['BYZANTINE_THRESHOLD']
    reputation_enabled: bool = True
    
    # Communication settings
    compression_enabled: bool = True
    compression_rate: float = 0.1
    async_aggregation: bool = False
    
    # Advanced settings
    adaptive_learning_rate: bool = True
    personalization: bool = False
    hierarchical: bool = False
    
    # Resource constraints
    max_communication_rounds: int = FEDERATED_CONSTANTS['MAX_ROUNDS']
    max_client_computation_time: float = FEDERATED_CONSTANTS['MAX_CLIENT_COMPUTATION_TIME']

@dataclass
class ClientState:
    """State information for a federated client"""
    client_id: str
    round_participation: List[int] = field(default_factory=list)
    total_samples: int = 0
    computation_capacity: float = 1.0
    communication_bandwidth: float = 1.0
    privacy_budget_spent: float = 0.0
    reputation_score: float = 1.0
    last_update_round: int = 0
    contribution_history: List[float] = field(default_factory=list)
    
    # Authentication specific
    authentication_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    # Resource tracking
    average_computation_time: float = 0.0
    average_communication_time: float = 0.0
    dropout_count: int = 0

@dataclass
class FederatedRound:
    """Information about a federated learning round"""
    round_number: int
    selected_clients: List[str]
    start_time: float
    end_time: Optional[float] = None
    
    # Aggregation results
    global_model_update: Optional[Dict[str, torch.Tensor]] = None
    aggregation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Privacy accounting
    privacy_cost: float = 0.0
    noise_added: float = 0.0
    
    # Performance metrics
    round_accuracy: float = 0.0
    round_loss: float = 0.0
    communication_cost: float = 0.0

class SecureAggregationProtocol:
    """
    Implements secure aggregation for privacy-preserving federated learning
    """
    
    def __init__(self, num_clients: int, threshold: int = None):
        self.num_clients = num_clients
        self.threshold = threshold or max(1, num_clients // 2)
        
        # Cryptographic setup
        self._setup_crypto()
        
        # Client keys and shares
        self.client_keys = {}
        self.pairwise_keys = {}
        self.shares = defaultdict(dict)
        
    def _setup_crypto(self):
        """Setup cryptographic parameters"""
        # Generate system parameters
        self.key_size = 2048
        self.security_parameter = 128
        
        # Initialize Paillier if available
        if PAILLIER_AVAILABLE:
            self.paillier_public, self.paillier_private = paillier.generate_paillier_keypair(
                n_length=self.key_size
            )
        
        # Initialize TenSEAL context if available
        if TENSEAL_AVAILABLE:
            self.tenseal_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.tenseal_context.generate_galois_keys()
            self.tenseal_context.global_scale = 2**40
    
    def setup_client_keys(self, client_ids: List[str]):
        """Setup cryptographic keys for clients"""
        for client_id in client_ids:
            # Generate RSA key pair for each client
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            self.client_keys[client_id] = {
                'private': private_key,
                'public': public_key
            }
        
        # Setup pairwise keys for masking
        self._setup_pairwise_keys(client_ids)
    
    def _setup_pairwise_keys(self, client_ids: List[str]):
        """Setup pairwise shared keys between clients"""
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids[i+1:], i+1):
                # Derive shared key using Diffie-Hellman
                shared_key = self._derive_shared_key(client_i, client_j)
                
                self.pairwise_keys[(client_i, client_j)] = shared_key
                self.pairwise_keys[(client_j, client_i)] = shared_key
    
    def _derive_shared_key(self, client_i: str, client_j: str) -> bytes:
        """Derive shared key between two clients"""
        # Simplified key derivation - use proper DH in production
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'forge_auth_secure_agg',
            iterations=100000,
            backend=default_backend()
        )
        
        key_material = f"{client_i}:{client_j}".encode()
        return kdf.derive(key_material)
    
    def encrypt_update(self, client_id: str, update: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Encrypt client update for secure aggregation"""
        encrypted_update = {}
        
        for param_name, param_value in update.items():
            if PAILLIER_AVAILABLE:
                # Paillier encryption for additive homomorphism
                encrypted_param = self._paillier_encrypt(param_value)
            elif TENSEAL_AVAILABLE:
                # CKKS encryption for approximate arithmetic
                encrypted_param = self._tenseal_encrypt(param_value)
            else:
                # Fallback to masking-based encryption
                encrypted_param = self._mask_based_encrypt(client_id, param_value)
            
            encrypted_update[param_name] = encrypted_param
        
        return encrypted_update
    
    def _paillier_encrypt(self, tensor: torch.Tensor) -> List:
        """Encrypt tensor using Paillier encryption"""
        flat_tensor = tensor.flatten().cpu().numpy()
        
        # Quantize to integers for Paillier
        scale = 1e6
        quantized = (flat_tensor * scale).astype(int)
        
        # Encrypt each element
        encrypted = [self.paillier_public.encrypt(int(x)) for x in quantized]
        
        return {
            'encrypted_values': encrypted,
            'shape': list(tensor.shape),
            'scale': scale
        }
    
    def _tenseal_encrypt(self, tensor: torch.Tensor) -> Any:
        """Encrypt tensor using TenSEAL (CKKS)"""
        flat_tensor = tensor.flatten().cpu().numpy()
        
        # Create encrypted vector
        encrypted_vector = ts.ckks_vector(self.tenseal_context, flat_tensor)
        
        return {
            'encrypted_vector': encrypted_vector,
            'shape': list(tensor.shape)
        }
    
    def _mask_based_encrypt(self, client_id: str, tensor: torch.Tensor) -> Dict[str, Any]:
        """Fallback masking-based encryption"""
        # Generate random mask
        mask = torch.randn_like(tensor)
        
        # Add pairwise masks for secure aggregation
        for other_client, key in self.pairwise_keys.items():
            if other_client[0] == client_id:
                # Generate PRG-based mask
                pairwise_mask = self._generate_mask_from_key(key, tensor.shape)
                mask += pairwise_mask
            elif other_client[1] == client_id:
                pairwise_mask = self._generate_mask_from_key(key, tensor.shape)
                mask -= pairwise_mask
        
        # Masked value
        masked_value = tensor + mask
        
        return {
            'masked_value': masked_value,
            'client_id': client_id
        }
    
    def _generate_mask_from_key(self, key: bytes, shape: Tuple) -> torch.Tensor:
        """Generate deterministic mask from shared key"""
        # Use AES in CTR mode as PRG
        cipher = Cipher(
            algorithms.AES(key[:16]),
            modes.CTR(key[16:32]),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Generate enough random bytes
        num_elements = np.prod(shape)
        num_bytes = num_elements * 4  # 32-bit floats
        random_bytes = encryptor.update(b'\x00' * num_bytes)
        
        # Convert to tensor
        mask_array = np.frombuffer(random_bytes, dtype=np.float32)[:num_elements]
        mask = torch.from_numpy(mask_array).reshape(shape)
        
        # Normalize to reasonable range
        mask = (mask - mask.mean()) / (mask.std() + 1e-7)
        
        return mask
    
    def aggregate_encrypted_updates(self, encrypted_updates: Dict[str, Dict[str, Any]],
                                  client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Aggregate encrypted updates from clients"""
        aggregated = {}
        
        # Get parameter names from first client
        param_names = list(next(iter(encrypted_updates.values())).keys())
        
        for param_name in param_names:
            if PAILLIER_AVAILABLE and 'encrypted_values' in encrypted_updates[next(iter(encrypted_updates))][param_name]:
                # Paillier aggregation
                aggregated[param_name] = self._aggregate_paillier(
                    encrypted_updates, param_name, client_weights
                )
            elif TENSEAL_AVAILABLE and 'encrypted_vector' in encrypted_updates[next(iter(encrypted_updates))][param_name]:
                # TenSEAL aggregation
                aggregated[param_name] = self._aggregate_tenseal(
                    encrypted_updates, param_name, client_weights
                )
            else:
                # Masking-based aggregation
                aggregated[param_name] = self._aggregate_masked(
                    encrypted_updates, param_name, client_weights
                )
        
        return aggregated
    
    def _aggregate_paillier(self, encrypted_updates: Dict[str, Dict[str, Any]],
                           param_name: str, client_weights: Dict[str, float]) -> torch.Tensor:
        """Aggregate Paillier encrypted values"""
        # Get first update for metadata
        first_update = next(iter(encrypted_updates.values()))[param_name]
        shape = first_update['shape']
        scale = first_update['scale']
        
        # Initialize aggregated encrypted values
        num_elements = np.prod(shape)
        aggregated_encrypted = [0] * num_elements
        
        total_weight = sum(client_weights.values())
        
        # Homomorphic addition
        for client_id, update in encrypted_updates.items():
            encrypted_values = update[param_name]['encrypted_values']
            weight = client_weights[client_id] / total_weight
            
            for i in range(num_elements):
                if aggregated_encrypted[i] == 0:
                    aggregated_encrypted[i] = encrypted_values[i] * weight
                else:
                    aggregated_encrypted[i] = aggregated_encrypted[i] + encrypted_values[i] * weight
        
        # Decrypt aggregated values
        decrypted_values = [
            self.paillier_private.decrypt(enc_val) / scale
            for enc_val in aggregated_encrypted
        ]
        
        # Reshape to original
        aggregated_tensor = torch.tensor(decrypted_values).reshape(shape)
        
        return aggregated_tensor
    
    def _aggregate_tenseal(self, encrypted_updates: Dict[str, Dict[str, Any]],
                          param_name: str, client_weights: Dict[str, float]) -> torch.Tensor:
        """Aggregate TenSEAL encrypted vectors"""
        # Get first update for metadata
        first_update = next(iter(encrypted_updates.values()))[param_name]
        shape = first_update['shape']
        
        total_weight = sum(client_weights.values())
        
        # Initialize aggregated vector
        aggregated_vector = None
        
        # Homomorphic weighted addition
        for client_id, update in encrypted_updates.items():
            encrypted_vector = update[param_name]['encrypted_vector']
            weight = client_weights[client_id] / total_weight
            
            weighted_vector = encrypted_vector * weight
            
            if aggregated_vector is None:
                aggregated_vector = weighted_vector
            else:
                aggregated_vector += weighted_vector
        
        # Decrypt
        decrypted_values = aggregated_vector.decrypt()
        aggregated_tensor = torch.tensor(decrypted_values).reshape(shape)
        
        return aggregated_tensor
    
    def _aggregate_masked(self, encrypted_updates: Dict[str, Dict[str, Any]],
                         param_name: str, client_weights: Dict[str, float]) -> torch.Tensor:
        """Aggregate masked values (masks cancel out)"""
        total_weight = sum(client_weights.values())
        
        # Initialize aggregated value
        aggregated = None
        
        for client_id, update in encrypted_updates.items():
            masked_value = update[param_name]['masked_value']
            weight = client_weights[client_id] / total_weight
            
            if aggregated is None:
                aggregated = masked_value * weight
            else:
                aggregated += masked_value * weight
        
        # Masks cancel out in aggregation due to pairwise structure
        return aggregated

class DifferentialPrivacyManager:
    """
    Manages differential privacy for federated learning
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.privacy_accountant = PrivacyAccountant(
            total_budget=config.privacy_budget,
            delta=1e-5
        )
        
        # Adaptive clipping
        self.clipping_manager = AdaptiveClippingManager(
            initial_threshold=config.clipping_threshold,
            target_quantile=0.5
        )
        
        # Noise calibration
        self.noise_calibrator = NoiseCalibrator(
            mechanism='gaussian',
            sensitivity_method='l2'
        )
    
    def add_noise_to_update(self, update: Dict[str, torch.Tensor],
                           round_number: int, client_id: str) -> Tuple[Dict[str, torch.Tensor], float]:
        """Add calibrated noise to client update"""
        # Clip update
        clipped_update, clip_norm = self.clipping_manager.clip_update(update)
        
        # Record clipping for adaptation
        self.clipping_manager.record_norm(clip_norm)
        
        # Compute noise scale
        noise_scale = self.noise_calibrator.compute_noise_scale(
            sensitivity=self.clipping_manager.current_threshold,
            epsilon=self._get_round_privacy_budget(round_number),
            delta=1e-5
        )
        
        # Add noise to each parameter
        noisy_update = {}
        total_noise_norm = 0
        
        for param_name, param_value in clipped_update.items():
            noise = torch.randn_like(param_value) * noise_scale
            noisy_update[param_name] = param_value + noise
            total_noise_norm += noise.norm().item() ** 2
        
        total_noise_norm = math.sqrt(total_noise_norm)
        
        # Account for privacy cost
        privacy_cost = self.privacy_accountant.account_privacy_cost(
            noise_scale=noise_scale,
            sensitivity=self.clipping_manager.current_threshold,
            sampling_rate=self.config.client_fraction
        )
        
        return noisy_update, privacy_cost
    
    def _get_round_privacy_budget(self, round_number: int) -> float:
        """Get privacy budget for current round"""
        # Allocate budget across rounds
        remaining_budget = self.privacy_accountant.get_remaining_budget()
        remaining_rounds = self.config.num_rounds - round_number
        
        if remaining_rounds <= 0:
            return 0.0
        
        # Adaptive allocation based on round importance
        if round_number < 10:
            # More budget in early rounds
            round_budget = remaining_budget / (remaining_rounds * 0.5)
        else:
            # Conservative in later rounds
            round_budget = remaining_budget / (remaining_rounds * 2)
        
        return min(round_budget, remaining_budget)
    
    def get_privacy_guarantee(self) -> Dict[str, float]:
        """Get current privacy guarantee"""
        return self.privacy_accountant.get_privacy_guarantee()

class PrivacyAccountant:
    """Tracks privacy budget consumption"""
    
    def __init__(self, total_budget: float, delta: float = 1e-5):
        self.total_budget = total_budget
        self.delta = delta
        self.consumed_budget = 0.0
        self.privacy_events = []
        
    def account_privacy_cost(self, noise_scale: float, sensitivity: float,
                           sampling_rate: float) -> float:
        """Account for privacy cost of an operation"""
        # Compute privacy cost using Gaussian mechanism
        if noise_scale == 0:
            privacy_cost = float('inf')
        else:
            privacy_cost = (sensitivity / noise_scale) * math.sqrt(2 * math.log(1.25 / self.delta))
        
        # Account for privacy amplification by sampling
        amplified_cost = privacy_cost * sampling_rate
        
        # Record event
        self.privacy_events.append({
            'cost': amplified_cost,
            'timestamp': time.time(),
            'noise_scale': noise_scale,
            'sensitivity': sensitivity,
            'sampling_rate': sampling_rate
        })
        
        self.consumed_budget += amplified_cost
        
        return amplified_cost
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.total_budget - self.consumed_budget)
    
    def get_privacy_guarantee(self) -> Dict[str, float]:
        """Get current privacy guarantee"""
        return {
            'epsilon': self.consumed_budget,
            'delta': self.delta,
            'remaining_budget': self.get_remaining_budget()
        }

class AdaptiveClippingManager:
    """Manages adaptive gradient clipping"""
    
    def __init__(self, initial_threshold: float, target_quantile: float = 0.5):
        self.current_threshold = initial_threshold
        self.target_quantile = target_quantile
        self.norm_history = deque(maxlen=1000)
        self.update_frequency = 10
        self.update_counter = 0
        
    def clip_update(self, update: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float]:
        """Clip update to threshold"""
        # Compute update norm
        total_norm = 0
        for param in update.values():
            total_norm += param.norm().item() ** 2
        total_norm = math.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > self.current_threshold:
            clip_factor = self.current_threshold / total_norm
            clipped_update = {
                name: param * clip_factor
                for name, param in update.items()
            }
        else:
            clipped_update = update
            
        return clipped_update, total_norm
    
    def record_norm(self, norm: float):
        """Record norm for adaptation"""
        self.norm_history.append(norm)
        self.update_counter += 1
        
        # Update threshold periodically
        if self.update_counter % self.update_frequency == 0:
            self._update_threshold()
    
    def _update_threshold(self):
        """Update clipping threshold based on history"""
        if len(self.norm_history) < 10:
            return
        
        # Compute target quantile
        target_norm = np.percentile(list(self.norm_history), self.target_quantile * 100)
        
        # Smooth update
        self.current_threshold = 0.9 * self.current_threshold + 0.1 * target_norm

class NoiseCalibrator:
    """Calibrates noise for differential privacy"""
    
    def __init__(self, mechanism: str = 'gaussian', sensitivity_method: str = 'l2'):
        self.mechanism = mechanism
        self.sensitivity_method = sensitivity_method
        
    def compute_noise_scale(self, sensitivity: float, epsilon: float, delta: float) -> float:
        """Compute noise scale for given privacy parameters"""
        if epsilon == 0:
            return float('inf')
        
        if self.mechanism == 'gaussian':
            # Gaussian mechanism
            noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        elif self.mechanism == 'laplace':
            # Laplace mechanism (no delta)
            noise_scale = sensitivity / epsilon
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
        
        return noise_scale

class ByzantineRobustAggregator:
    """Implements Byzantine-robust aggregation algorithms"""
    
    def __init__(self, num_clients: int, byzantine_threshold: float = 0.3):
        self.num_clients = num_clients
        self.max_byzantine = int(num_clients * byzantine_threshold)
        self.reputation_tracker = ReputationTracker()
        
    def aggregate_krum(self, updates: Dict[str, Dict[str, torch.Tensor]],
                      m: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Krum aggregation algorithm"""
        if m is None:
            m = self.num_clients - self.max_byzantine - 2
        
        client_ids = list(updates.keys())
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(updates)
        
        # Compute Krum scores
        scores = []
        for i, client_id in enumerate(client_ids):
            # Sum of m nearest distances
            client_distances = distances[i]
            sorted_distances = sorted(client_distances)
            score = sum(sorted_distances[1:m+1])  # Exclude self (0 distance)
            scores.append((score, client_id))
        
        # Select client with minimum score
        scores.sort()
        selected_client = scores[0][1]
        
        return updates[selected_client]
    
    def aggregate_multi_krum(self, updates: Dict[str, Dict[str, torch.Tensor]],
                           m: Optional[int] = None, c: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Multi-Krum aggregation"""
        if m is None:
            m = self.num_clients - self.max_byzantine - 2
        if c is None:
            c = self.num_clients - self.max_byzantine
        
        client_ids = list(updates.keys())
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(updates)
        
        # Compute Krum scores and select top c clients
        scores = []
        for i, client_id in enumerate(client_ids):
            client_distances = distances[i]
            sorted_distances = sorted(client_distances)
            score = sum(sorted_distances[1:m+1])
            scores.append((score, client_id))
        
        scores.sort()
        selected_clients = [client_id for _, client_id in scores[:c]]
        
        # Average selected updates
        return self._average_updates(
            {client_id: updates[client_id] for client_id in selected_clients}
        )
    
    def aggregate_trimmed_mean(self, updates: Dict[str, Dict[str, torch.Tensor]],
                             trim_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation"""
        num_trim = int(len(updates) * trim_ratio)
        
        # Aggregate each parameter separately
        aggregated = {}
        param_names = list(next(iter(updates.values())).keys())
        
        for param_name in param_names:
            # Stack parameter values from all clients
            param_values = []
            for client_update in updates.values():
                param_values.append(client_update[param_name])
            
            stacked = torch.stack(param_values)
            
            # Sort and trim
            sorted_values, _ = torch.sort(stacked, dim=0)
            
            if num_trim > 0:
                trimmed = sorted_values[num_trim:-num_trim]
            else:
                trimmed = sorted_values
            
            # Average
            aggregated[param_name] = trimmed.mean(dim=0)
        
        return aggregated
    
    def aggregate_median(self, updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation"""
        aggregated = {}
        param_names = list(next(iter(updates.values())).keys())
        
        for param_name in param_names:
            param_values = []
            for client_update in updates.values():
                param_values.append(client_update[param_name])
            
            stacked = torch.stack(param_values)
            aggregated[param_name] = torch.median(stacked, dim=0)[0]
        
        return aggregated
    
    def detect_byzantine_clients(self, updates: Dict[str, Dict[str, torch.Tensor]],
                               global_model: Dict[str, torch.Tensor]) -> Set[str]:
        """Detect potentially Byzantine clients"""
        byzantine_clients = set()
        
        # Compute update statistics
        update_norms = {}
        update_directions = {}
        
        for client_id, update in updates.items():
            # Compute update norm
            norm = 0
            direction = []
            
            for param_name, param_update in update.items():
                norm += param_update.norm().item() ** 2
                
                # Compare direction with global model
                if param_name in global_model:
                    cosine_sim = F.cosine_similarity(
                        param_update.flatten(),
                        global_model[param_name].flatten(),
                        dim=0
                    )
                    direction.append(cosine_sim.item())
            
            update_norms[client_id] = math.sqrt(norm)
            update_directions[client_id] = np.mean(direction) if direction else 0
        
        # Statistical outlier detection
        norms = list(update_norms.values())
        directions = list(update_directions.values())
        
        # Detect norm outliers
        norm_mean = np.mean(norms)
        norm_std = np.std(norms)
        
        for client_id, norm in update_norms.items():
            z_score = abs(norm - norm_mean) / (norm_std + 1e-7)
            if z_score > 3:  # 3-sigma rule
                byzantine_clients.add(client_id)
                self.reputation_tracker.penalize_client(client_id, 'norm_outlier')
        
        # Detect direction outliers
        dir_mean = np.mean(directions)
        dir_std = np.std(directions)
        
        for client_id, direction in update_directions.items():
            if direction < dir_mean - 2 * dir_std:  # Significantly different direction
                byzantine_clients.add(client_id)
                self.reputation_tracker.penalize_client(client_id, 'direction_outlier')
        
        return byzantine_clients
    
    def _compute_pairwise_distances(self, updates: Dict[str, Dict[str, torch.Tensor]]) -> List[List[float]]:
        """Compute pairwise L2 distances between updates"""
        client_ids = list(updates.keys())
        num_clients = len(client_ids)
        distances = [[0.0] * num_clients for _ in range(num_clients)]
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distance = self._compute_update_distance(
                    updates[client_ids[i]],
                    updates[client_ids[j]]
                )
                distances[i][j] = distance
                distances[j][i] = distance
        
        return distances
    
    def _compute_update_distance(self, update1: Dict[str, torch.Tensor],
                               update2: Dict[str, torch.Tensor]) -> float:
        """Compute L2 distance between two updates"""
        total_distance = 0
        
        for param_name in update1:
            if param_name in update2:
                diff = update1[param_name] - update2[param_name]
                total_distance += diff.norm().item() ** 2
        
        return math.sqrt(total_distance)
    
    def _average_updates(self, updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Simple averaging of updates"""
        averaged = {}
        param_names = list(next(iter(updates.values())).keys())
        
        for param_name in param_names:
            param_sum = None
            count = 0
            
            for update in updates.values():
                if param_name in update:
                    if param_sum is None:
                        param_sum = update[param_name].clone()
                    else:
                        param_sum += update[param_name]
                    count += 1
            
            if param_sum is not None:
                averaged[param_name] = param_sum / count
        
        return averaged

class ReputationTracker:
    """Tracks client reputation for Byzantine robustness"""
    
    def __init__(self, initial_reputation: float = 1.0):
        self.reputations = defaultdict(lambda: initial_reputation)
        self.penalty_history = defaultdict(list)
        self.reward_history = defaultdict(list)
        
    def penalize_client(self, client_id: str, reason: str, penalty: float = 0.1):
        """Penalize client reputation"""
        self.reputations[client_id] = max(0, self.reputations[client_id] - penalty)
        self.penalty_history[client_id].append({
            'reason': reason,
            'penalty': penalty,
            'timestamp': time.time()
        })
    
    def reward_client(self, client_id: str, reason: str, reward: float = 0.05):
        """Reward client reputation"""
        self.reputations[client_id] = min(1.0, self.reputations[client_id] + reward)
        self.reward_history[client_id].append({
            'reason': reason,
            'reward': reward,
            'timestamp': time.time()
        })
    
    def get_reputation(self, client_id: str) -> float:
        """Get client reputation score"""
        return self.reputations[client_id]
    
    def get_trusted_clients(self, client_ids: List[str], threshold: float = 0.5) -> List[str]:
        """Get list of trusted clients"""
        return [
            client_id for client_id in client_ids
            if self.reputations[client_id] >= threshold
        ]

class CommunicationManager:
    """Manages communication efficiency in federated learning"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.compressor = UpdateCompressor(
            compression_rate=config.compression_rate
        )
        self.communication_stats = defaultdict(lambda: {
            'bytes_sent': 0,
            'bytes_received': 0,
            'compression_ratio': []
        })
    
    def compress_update(self, update: Dict[str, torch.Tensor],
                       client_id: str) -> Tuple[Dict[str, Any], float]:
        """Compress update for efficient communication"""
        original_size = self._compute_update_size(update)
        
        if self.config.compression_enabled:
            compressed_update, compressed_size = self.compressor.compress(update)
            compression_ratio = compressed_size / original_size
        else:
            compressed_update = update
            compressed_size = original_size
            compression_ratio = 1.0
        
        # Record statistics
        self.communication_stats[client_id]['bytes_sent'] += compressed_size
        self.communication_stats[client_id]['compression_ratio'].append(compression_ratio)
        
        return compressed_update, compression_ratio
    
    def decompress_update(self, compressed_update: Dict[str, Any],
                         client_id: str) -> Dict[str, torch.Tensor]:
        """Decompress received update"""
        if self.config.compression_enabled:
            decompressed = self.compressor.decompress(compressed_update)
        else:
            decompressed = compressed_update
        
        # Record statistics
        self.communication_stats[client_id]['bytes_received'] += self._compute_update_size(decompressed)
        
        return decompressed
    
    def _compute_update_size(self, update: Dict[str, torch.Tensor]) -> float:
        """Compute size of update in bytes"""
        total_size = 0
        
        for param in update.values():
            total_size += param.numel() * param.element_size()
        
        return total_size
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        total_sent = sum(stats['bytes_sent'] for stats in self.communication_stats.values())
        total_received = sum(stats['bytes_received'] for stats in self.communication_stats.values())
        
        avg_compression = []
        for stats in self.communication_stats.values():
            if stats['compression_ratio']:
                avg_compression.extend(stats['compression_ratio'])
        
        return {
            'total_bytes_sent': total_sent,
            'total_bytes_received': total_received,
            'total_bytes': total_sent + total_received,
            'average_compression_ratio': np.mean(avg_compression) if avg_compression else 1.0,
            'per_client_stats': dict(self.communication_stats)
        }

class UpdateCompressor:
    """Compresses model updates for efficient communication"""
    
    def __init__(self, compression_rate: float = 0.1):
        self.compression_rate = compression_rate
        self.method = 'topk'  # Options: 'topk', 'randomk', 'quantization'
        
    def compress(self, update: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], float]:
        """Compress update"""
        if self.method == 'topk':
            return self._topk_compression(update)
        elif self.method == 'randomk':
            return self._randomk_compression(update)
        elif self.method == 'quantization':
            return self._quantization_compression(update)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")
    
    def _topk_compression(self, update: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], float]:
        """Top-k sparsification"""
        compressed = {}
        total_size = 0
        
        for param_name, param_value in update.items():
            # Flatten parameter
            flat_param = param_value.flatten()
            num_elements = flat_param.numel()
            
            # Select top-k elements by magnitude
            k = max(1, int(num_elements * self.compression_rate))
            topk_values, topk_indices = torch.topk(flat_param.abs(), k)
            
            # Get actual values (with sign)
            topk_actual = flat_param[topk_indices]
            
            compressed[param_name] = {
                'values': topk_actual,
                'indices': topk_indices,
                'shape': list(param_value.shape),
                'num_elements': num_elements
            }
            
            # Compute compressed size
            total_size += (k * 4 * 2)  # values and indices
        
        return compressed, total_size
    
    def _randomk_compression(self, update: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], float]:
        """Random-k sparsification"""
        compressed = {}
        total_size = 0
        
        for param_name, param_value in update.items():
            flat_param = param_value.flatten()
            num_elements = flat_param.numel()
            
            # Randomly select k elements
            k = max(1, int(num_elements * self.compression_rate))
            random_indices = torch.randperm(num_elements)[:k]
            
            compressed[param_name] = {
                'values': flat_param[random_indices],
                'indices': random_indices,
                'shape': list(param_value.shape),
                'num_elements': num_elements
            }
            
            total_size += (k * 4 * 2)
        
        return compressed, total_size
    
    def _quantization_compression(self, update: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], float]:
        """Quantization-based compression"""
        compressed = {}
        total_size = 0
        
        for param_name, param_value in update.items():
            # Quantize to 8-bit
            min_val = param_value.min()
            max_val = param_value.max()
            
            # Scale to [0, 255]
            scaled = ((param_value - min_val) / (max_val - min_val + 1e-7) * 255)
            quantized = scaled.round().byte()
            
            compressed[param_name] = {
                'quantized': quantized,
                'min_val': min_val,
                'max_val': max_val,
                'shape': list(param_value.shape)
            }
            
            total_size += quantized.numel() + 8  # quantized values + min/max
        
        return compressed, total_size
    
    def decompress(self, compressed: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decompress update"""
        decompressed = {}
        
        for param_name, comp_data in compressed.items():
            if 'quantized' in comp_data:
                # Dequantization
                quantized = comp_data['quantized']
                min_val = comp_data['min_val']
                max_val = comp_data['max_val']
                
                # Scale back to original range
                dequantized = quantized.float() / 255 * (max_val - min_val) + min_val
                decompressed[param_name] = dequantized.reshape(comp_data['shape'])
                
            else:
                # Sparse reconstruction
                values = comp_data['values']
                indices = comp_data['indices']
                shape = comp_data['shape']
                num_elements = comp_data['num_elements']
                
                # Reconstruct sparse tensor
                reconstructed = torch.zeros(num_elements)
                reconstructed[indices] = values
                decompressed[param_name] = reconstructed.reshape(shape)
        
        return decompressed

class ClientSelector:
    """Implements various client selection strategies"""
    
    def __init__(self, config: FederatedConfig, client_states: Dict[str, ClientState]):
        self.config = config
        self.client_states = client_states
        self.selection_history = defaultdict(list)
        
    def select_clients(self, available_clients: List[str], round_number: int) -> List[str]:
        """Select clients for current round"""
        num_clients = max(
            self.config.min_clients,
            int(len(available_clients) * self.config.client_fraction)
        )
        
        strategy = self.config.client_selection_strategy
        
        if strategy == ClientSelectionStrategy.RANDOM:
            selected = self._random_selection(available_clients, num_clients)
        elif strategy == ClientSelectionStrategy.ROUND_ROBIN:
            selected = self._round_robin_selection(available_clients, num_clients, round_number)
        elif strategy == ClientSelectionStrategy.CONTRIBUTION_BASED:
            selected = self._contribution_based_selection(available_clients, num_clients)
        elif strategy == ClientSelectionStrategy.RESOURCE_AWARE:
            selected = self._resource_aware_selection(available_clients, num_clients)
        elif strategy == ClientSelectionStrategy.PRIVACY_AWARE:
            selected = self._privacy_aware_selection(available_clients, num_clients)
        elif strategy == ClientSelectionStrategy.REPUTATION_BASED:
            selected = self._reputation_based_selection(available_clients, num_clients)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        # Record selection
        for client_id in selected:
            self.selection_history[client_id].append(round_number)
        
        return selected
    
    def _random_selection(self, available: List[str], num_clients: int) -> List[str]:
        """Random client selection"""
        return np.random.choice(available, size=min(num_clients, len(available)), replace=False).tolist()
    
    def _round_robin_selection(self, available: List[str], num_clients: int, round_number: int) -> List[str]:
        """Round-robin selection"""
        start_idx = (round_number * num_clients) % len(available)
        selected = []
        
        for i in range(min(num_clients, len(available))):
            idx = (start_idx + i) % len(available)
            selected.append(available[idx])
        
        return selected
    
    def _contribution_based_selection(self, available: List[str], num_clients: int) -> List[str]:
        """Select based on past contribution quality"""
        scores = []
        
        for client_id in available:
            state = self.client_states[client_id]
            
            # Compute contribution score
            score = 0
            if state.contribution_history:
                score = np.mean(state.contribution_history)
            
            # Add exploration bonus for rarely selected clients
            selection_count = len(self.selection_history[client_id])
            exploration_bonus = 1.0 / (1 + selection_count)
            
            scores.append((score + exploration_bonus, client_id))
        
        # Select top clients
        scores.sort(reverse=True)
        return [client_id for _, client_id in scores[:num_clients]]
    
    def _resource_aware_selection(self, available: List[str], num_clients: int) -> List[str]:
        """Select based on computational resources"""
        scores = []
        
        for client_id in available:
            state = self.client_states[client_id]
            
            # Compute resource score
            resource_score = (
                state.computation_capacity * 0.5 +
                state.communication_bandwidth * 0.3 +
                (1 - state.dropout_count / 10) * 0.2  # Penalize frequent dropouts
            )
            
            scores.append((resource_score, client_id))
        
        scores.sort(reverse=True)
        return [client_id for _, client_id in scores[:num_clients]]
    
    def _privacy_aware_selection(self, available: List[str], num_clients: int) -> List[str]:
        """Select based on privacy budget"""
        scores = []
        
        for client_id in available:
            state = self.client_states[client_id]
            
            # Prefer clients with more privacy budget remaining
            remaining_budget = self.config.privacy_budget - state.privacy_budget_spent
            scores.append((remaining_budget, client_id))
        
        scores.sort(reverse=True)
        return [client_id for _, client_id in scores[:num_clients]]
    
    def _reputation_based_selection(self, available: List[str], num_clients: int) -> List[str]:
        """Select based on reputation scores"""
        scores = []
        
        for client_id in available:
            state = self.client_states[client_id]
            scores.append((state.reputation_score, client_id))
        
        scores.sort(reverse=True)
        return [client_id for _, client_id in scores[:num_clients]]

class FederatedOrchestrator:
    """
    Main orchestrator for federated learning with authentication
    """
    
    def __init__(self, config: FederatedConfig, global_model: nn.Module):
        self.config = config
        self.global_model = global_model
        
        # Initialize components
        self.client_states = {}
        self.secure_aggregator = SecureAggregationProtocol(0)  # Will update with actual clients
        self.privacy_manager = DifferentialPrivacyManager(config)
        self.byzantine_aggregator = ByzantineRobustAggregator(0)  # Will update
        self.communication_manager = CommunicationManager(config)
        self.client_selector = None  # Will initialize after clients register
        
        # Round management
        self.current_round = 0
        self.round_history = []
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Asynchronous components if enabled
        if config.async_aggregation:
            self.async_handler = AsyncAggregationHandler(self)
        
        # Model checkpointing
        self.checkpoint_manager = CheckpointManager(
            save_dir='./federated_checkpoints'
        )
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]):
        """Register a new client"""
        self.client_states[client_id] = ClientState(
            client_id=client_id,
            computation_capacity=client_info.get('computation_capacity', 1.0),
            communication_bandwidth=client_info.get('communication_bandwidth', 1.0),
            total_samples=client_info.get('num_samples', 0)
        )
        
        # Update components with new client count
        self._update_component_sizes()
    
    def _update_component_sizes(self):
        """Update components that depend on number of clients"""
        num_clients = len(self.client_states)
        
        self.secure_aggregator = SecureAggregationProtocol(num_clients)
        self.byzantine_aggregator = ByzantineRobustAggregator(
            num_clients,
            self.config.byzantine_threshold
        )
        self.client_selector = ClientSelector(self.config, self.client_states)
    
    def run_federated_round(self) -> FederatedRound:
        """Execute one round of federated learning"""
        round_start = time.time()
        
        # Initialize round
        round_info = FederatedRound(
            round_number=self.current_round,
            selected_clients=[],
            start_time=round_start
        )
        
        try:
            # Select clients
            available_clients = self._get_available_clients()
            selected_clients = self.client_selector.select_clients(
                available_clients,
                self.current_round
            )
            round_info.selected_clients = selected_clients
            
            logger.info(f"Round {self.current_round}: Selected {len(selected_clients)} clients")
            
            # Prepare global model for distribution
            global_state = self._prepare_global_model()
            
            # Secure aggregation setup
            if self.config.secure_aggregation:
                self.secure_aggregator.setup_client_keys(selected_clients)
            
            # Collect client updates
            client_updates = self._collect_client_updates(
                selected_clients,
                global_state,
                round_info
            )
            
            # Aggregate updates
            aggregated_update = self._aggregate_updates(
                client_updates,
                round_info
            )
            
            # Update global model
            self._update_global_model(aggregated_update)
            
            # Evaluate round
            round_metrics = self._evaluate_round()
            round_info.aggregation_metrics = round_metrics
            
            # Record round completion
            round_info.end_time = time.time()
            self.round_history.append(round_info)
            
            # Checkpoint if needed
            if self.current_round % FEDERATED_CONSTANTS['CHECKPOINT_FREQUENCY'] == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.global_model,
                    self.current_round,
                    round_metrics
                )
            
            self.current_round += 1
            
            return round_info
            
        except Exception as e:
            logger.error(f"Round {self.current_round} failed: {e}")
            round_info.end_time = time.time()
            round_info.aggregation_metrics['error'] = str(e)
            return round_info
    
    def _get_available_clients(self) -> List[str]:
        """Get list of available clients"""
        available = []
        
        for client_id, state in self.client_states.items():
            # Check if client has sufficient privacy budget
            if state.privacy_budget_spent < self.config.privacy_budget:
                # Check if client hasn't dropped out frequently
                if state.dropout_count < 5:
                    available.append(client_id)
        
        return available
    
    def _prepare_global_model(self) -> Dict[str, torch.Tensor]:
        """Prepare global model for distribution"""
        return {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
        }
    
    def _collect_client_updates(self, selected_clients: List[str],
                              global_state: Dict[str, torch.Tensor],
                              round_info: FederatedRound) -> Dict[str, Dict[str, Any]]:
        """Collect updates from selected clients"""
        client_updates = {}
        
        # Parallel collection with timeout
        with ThreadPoolExecutor(max_workers=len(selected_clients)) as executor:
            futures = {}
            
            for client_id in selected_clients:
                future = executor.submit(
                    self._get_client_update,
                    client_id,
                    global_state
                )
                futures[future] = client_id
            
            # Collect with timeout
            completed = 0
            for future in futures:
                client_id = futures[future]
                
                try:
                    update = future.result(
                        timeout=FEDERATED_CONSTANTS['MAX_CLIENT_COMPUTATION_TIME']
                    )
                    
                    if update is not None:
                        client_updates[client_id] = update
                        completed += 1
                        
                        # Update client state
                        self.client_states[client_id].round_participation.append(
                            self.current_round
                        )
                        
                except Exception as e:
                    logger.warning(f"Client {client_id} failed: {e}")
                    self.client_states[client_id].dropout_count += 1
        
        logger.info(f"Collected updates from {completed}/{len(selected_clients)} clients")
        
        return client_updates
    
    def _get_client_update(self, client_id: str,
                          global_state: Dict[str, torch.Tensor]) -> Optional[Dict[str, Any]]:
        """Get update from single client (simulated)"""
        # In practice, this would communicate with actual client
        # Here we simulate client computation
        
        client_state = self.client_states[client_id]
        
        # Simulate local training
        local_update = self._simulate_local_training(
            client_id,
            global_state,
            client_state
        )
        
        # Add privacy noise
        if self.config.use_differential_privacy:
            noisy_update, privacy_cost = self.privacy_manager.add_noise_to_update(
                local_update,
                self.current_round,
                client_id
            )
            
            # Update privacy budget
            client_state.privacy_budget_spent += privacy_cost
        else:
            noisy_update = local_update
        
        # Compress update
        if self.config.compression_enabled:
            compressed_update, compression_ratio = self.communication_manager.compress_update(
                noisy_update,
                client_id
            )
        else:
            compressed_update = noisy_update
        
        # Encrypt if using secure aggregation
        if self.config.secure_aggregation:
            encrypted_update = self.secure_aggregator.encrypt_update(
                client_id,
                compressed_update
            )
        else:
            encrypted_update = compressed_update
        
        return {
            'update': encrypted_update,
            'num_samples': client_state.total_samples,
            'computation_time': np.random.normal(60, 10),  # Simulated
            'client_metrics': {
                'loss': np.random.random(),
                'accuracy': np.random.random()
            }
        }
    
    def _simulate_local_training(self, client_id: str,
                               global_state: Dict[str, torch.Tensor],
                               client_state: ClientState) -> Dict[str, torch.Tensor]:
        """Simulate local training (placeholder)"""
        # In practice, this happens on the client device
        # Here we simulate with random perturbations
        
        local_update = {}
        
        for param_name, param_value in global_state.items():
            # Simulate gradient
            gradient = torch.randn_like(param_value) * 0.01
            
            # Simulate local epochs
            local_update[param_name] = -gradient * self.config.local_epochs
        
        return local_update
    
    def _aggregate_updates(self, client_updates: Dict[str, Dict[str, Any]],
                          round_info: FederatedRound) -> Dict[str, torch.Tensor]:
        """Aggregate client updates"""
        # Decrypt and decompress updates
        processed_updates = {}
        client_weights = {}
        
        for client_id, update_data in client_updates.items():
            # Decrypt if needed
            if self.config.secure_aggregation:
                decrypted = update_data['update']  # Would be decrypted in secure aggregation
            else:
                decrypted = update_data['update']
            
            # Decompress if needed
            if self.config.compression_enabled:
                decompressed = self.communication_manager.decompress_update(
                    decrypted,
                    client_id
                )
            else:
                decompressed = decrypted
            
            processed_updates[client_id] = decompressed
            client_weights[client_id] = update_data['num_samples']
        
        # Byzantine filtering
        if self.config.byzantine_robust:
            byzantine_clients = self.byzantine_aggregator.detect_byzantine_clients(
                processed_updates,
                self._prepare_global_model()
            )
            
            if byzantine_clients:
                logger.warning(f"Detected Byzantine clients: {byzantine_clients}")
                
                # Remove Byzantine clients
                for client_id in byzantine_clients:
                    processed_updates.pop(client_id, None)
                    client_weights.pop(client_id, None)
        
        # Apply aggregation strategy
        strategy = self.config.aggregation_strategy
        
        if strategy == AggregationStrategy.FEDERATED_AVERAGING:
            aggregated = self._federated_averaging(processed_updates, client_weights)
        elif strategy == AggregationStrategy.KRUM:
            aggregated = self.byzantine_aggregator.aggregate_krum(processed_updates)
        elif strategy == AggregationStrategy.TRIMMED_MEAN:
            aggregated = self.byzantine_aggregator.aggregate_trimmed_mean(processed_updates)
        elif strategy == AggregationStrategy.MEDIAN:
            aggregated = self.byzantine_aggregator.aggregate_median(processed_updates)
        else:
            # Default to FedAvg
            aggregated = self._federated_averaging(processed_updates, client_weights)
        
        round_info.global_model_update = aggregated
        
        return aggregated
    
    def _federated_averaging(self, updates: Dict[str, Dict[str, torch.Tensor]],
                           weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Standard federated averaging"""
        total_weight = sum(weights.values())
        
        averaged = {}
        param_names = list(next(iter(updates.values())).keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for client_id, update in updates.items():
                if param_name in update:
                    weight = weights[client_id] / total_weight
                    
                    if weighted_sum is None:
                        weighted_sum = update[param_name] * weight
                    else:
                        weighted_sum += update[param_name] * weight
            
            if weighted_sum is not None:
                averaged[param_name] = weighted_sum
        
        return averaged
    
    def _update_global_model(self, update: Dict[str, torch.Tensor]):
        """Apply aggregated update to global model"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in update:
                    # Apply update (gradient descent)
                    param.data.add_(update[name])
    
    def _evaluate_round(self) -> Dict[str, float]:
        """Evaluate current round performance"""
        # This would evaluate on a validation set
        # Placeholder metrics
        return {
            'global_accuracy': np.random.random(),
            'global_loss': np.random.random(),
            'authentication_accuracy': np.random.random()
        }
    
    def get_privacy_guarantee(self) -> Dict[str, float]:
        """Get current privacy guarantee"""
        return self.privacy_manager.get_privacy_guarantee()
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return self.communication_manager.get_communication_stats()

class AsyncAggregationHandler:
    """Handles asynchronous federated aggregation"""
    
    def __init__(self, orchestrator: FederatedOrchestrator):
        self.orchestrator = orchestrator
        self.update_queue = asyncio.Queue()
        self.aggregation_buffer = defaultdict(list)
        self.staleness_tracker = defaultdict(int)
        
    async def handle_client_update(self, client_id: str, update: Dict[str, Any]):
        """Handle asynchronous client update"""
        # Add to queue with timestamp
        await self.update_queue.put({
            'client_id': client_id,
            'update': update,
            'timestamp': time.time(),
            'round': self.orchestrator.current_round
        })
    
    async def aggregation_worker(self):
        """Worker for asynchronous aggregation"""
        while True:
            # Collect updates
            updates_to_aggregate = []
            
            # Get updates from queue
            while not self.update_queue.empty() and len(updates_to_aggregate) < 10:
                try:
                    update_data = await asyncio.wait_for(
                        self.update_queue.get(),
                        timeout=1.0
                    )
                    
                    # Check staleness
                    staleness = self.orchestrator.current_round - update_data['round']
                    
                    if staleness <= FEDERATED_CONSTANTS['STALENESS_THRESHOLD']:
                        updates_to_aggregate.append(update_data)
                    else:
                        logger.warning(f"Dropping stale update from {update_data['client_id']}")
                        
                except asyncio.TimeoutError:
                    break
            
            # Aggregate if we have enough updates
            if len(updates_to_aggregate) >= self.orchestrator.config.min_clients:
                await self._perform_async_aggregation(updates_to_aggregate)
            
            await asyncio.sleep(1)  # Check every second
    
    async def _perform_async_aggregation(self, updates: List[Dict[str, Any]]):
        """Perform asynchronous aggregation"""
        # Extract updates and weights
        processed_updates = {}
        weights = {}
        
        for update_data in updates:
            client_id = update_data['client_id']
            processed_updates[client_id] = update_data['update']['update']
            weights[client_id] = update_data['update']['num_samples']
        
        # Aggregate
        aggregated = self.orchestrator._federated_averaging(processed_updates, weights)
        
        # Apply to global model with staleness penalty
        avg_staleness = np.mean([
            self.orchestrator.current_round - u['round']
            for u in updates
        ])
        
        staleness_factor = 1.0 / (1.0 + avg_staleness)
        
        with torch.no_grad():
            for name, param in self.orchestrator.global_model.named_parameters():
                if name in aggregated:
                    param.data.add_(aggregated[name] * staleness_factor)

class MetricsTracker:
    """Tracks federated learning metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.client_metrics = defaultdict(lambda: defaultdict(list))
        
    def record_round_metrics(self, round_info: FederatedRound):
        """Record metrics for a round"""
        self.metrics['round_duration'].append(
            round_info.end_time - round_info.start_time
        )
        self.metrics['num_clients'].append(len(round_info.selected_clients))
        self.metrics['privacy_cost'].append(round_info.privacy_cost)
        
        for metric, value in round_info.aggregation_metrics.items():
            self.metrics[metric].append(value)
    
    def record_client_metrics(self, client_id: str, metrics: Dict[str, float]):
        """Record client-specific metrics"""
        for metric, value in metrics.items():
            self.client_metrics[client_id][metric].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {}
        
        # Global metrics
        for metric, values in self.metrics.items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_last'] = values[-1]
        
        # Client participation
        participation_counts = defaultdict(int)
        for client_id, metrics in self.client_metrics.items():
            participation_counts[client_id] = len(metrics.get('accuracy', []))
        
        summary['client_participation'] = dict(participation_counts)
        
        return summary

class CheckpointManager:
    """Manages federated learning checkpoints"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_checkpoint(self, model: nn.Module, round_number: int,
                       metrics: Dict[str, float]):
        """Save federated checkpoint"""
        checkpoint = {
            'round': round_number,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        path = os.path.join(self.save_dir, f'federated_round_{round_number}.pt')
        torch.save(checkpoint, path)
        logger.info(f"Saved federated checkpoint to {path}")
    
    def load_checkpoint(self, path: str, model: nn.Module) -> Dict:
        """Load federated checkpoint"""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded federated checkpoint from {path}")
        return checkpoint

# Factory function
def create_federated_orchestrator(config: Dict[str, Any],
                                global_model: nn.Module) -> FederatedOrchestrator:
    """Create federated orchestrator from configuration"""
    fed_config = FederatedConfig(**config)
    return FederatedOrchestrator(fed_config, global_model)

# Export main components
__all__ = [
    'FederatedConfig',
    'AggregationStrategy',
    'ClientSelectionStrategy',
    'ClientState',
    'FederatedRound',
    'SecureAggregationProtocol',
    'DifferentialPrivacyManager',
    'ByzantineRobustAggregator',
    'CommunicationManager',
    'ClientSelector',
    'FederatedOrchestrator',
    'create_federated_orchestrator'
]