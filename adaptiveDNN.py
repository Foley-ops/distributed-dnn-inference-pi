#!/usr/bin/env python3

import os
os.environ['ATEN_CPU_CAPABILITY'] = ''

import time
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
from dotenv import load_dotenv
import argparse
import logging
import socket
import sys
from typing import List, Dict, Tuple
import psutil
import paramiko
import threading
import queue
import random
from collections import deque

# Load environment variables
load_dotenv()

# Environment variables with defaults
SSH_USERNAME = os.getenv('SSH_USERNAME', 'cc')
SSH_KEY_PATH = os.getenv('SSH_KEY_PATH', os.path.expanduser('~/.ssh/id_ed25519'))
MASTER_INTERFACE = os.getenv('MASTER_INTERFACE', 'enp6s0')
WORKER_INTERFACE = os.getenv('WORKER_INTERFACE', 'eth0')
MODEL_DIR = os.getenv('MODEL_DIR', './')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s [%(hostname)s:rank%(rank)s]',
)

logger = logging.getLogger(__name__)

# Device health metrics thresholds
LATENCY_THRESHOLD = 50  # ms
CPU_USAGE_THRESHOLD = 80  # %
MEMORY_USAGE_THRESHOLD = 90  # %
INFERENCE_TIME_THRESHOLD = 2.0  # seconds

class DeviceMetrics:
    """Stores metrics for a device"""
    def __init__(self):
        self.ping_ms = 0.0
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.inference_times = deque(maxlen=10)
        self.last_update = time.time()
        self.p2p_latencies = {}  # {peer_worker: latency_ms}

class DeviceMonitor:
    """Monitors device health metrics and P2P connectivity"""
    def __init__(self, workers, ssh_username, ssh_key_path):
        self.workers = workers
        self.ssh_username = ssh_username
        self.ssh_key_path = ssh_key_path
        self.metrics = {worker: DeviceMetrics() for worker in workers}
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Continuously monitor device metrics"""
        while self.running:
            for worker in self.workers:
                try:
                    self._update_device_metrics(worker)
                except Exception as e:
                    logger.error(f"Error monitoring {worker}: {e}")
            
            # Periodically check P2P connectivity
            try:
                self._update_p2p_metrics()
            except Exception as e:
                logger.error(f"Error checking P2P connectivity: {e}")
                
            time.sleep(5)  # Monitor every 5 seconds

    def _update_device_metrics(self, worker):
        """Update metrics for a specific device"""
        # Get ping latency
        ping_cmd = f"ping -c 1 -W 1 {worker}"
        result = os.popen(ping_cmd).read()
        if 'time=' in result:
            ping_time = float(result.split('time=')[1].split(' ')[0])
            self.metrics[worker].ping_ms = ping_time

        # Get CPU and memory via SSH
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=worker,
                username=self.ssh_username,
                key_filename=self.ssh_key_path,
                timeout=5
            )
            
            # Get CPU usage
            stdin, stdout, stderr = client.exec_command("grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage}'")
            cpu_usage = float(stdout.read().decode().strip())
            
            # Get memory usage
            stdin, stdout, stderr = client.exec_command("free | grep Mem | awk '{print $3/$2 * 100.0}'")
            memory_usage = float(stdout.read().decode().strip())
            
            client.close()
            
            self.metrics[worker].cpu_percent = cpu_usage
            self.metrics[worker].memory_percent = memory_usage
            self.metrics[worker].last_update = time.time()
            
        except Exception as e:
            logger.error(f"SSH error for {worker}: {e}")

    def _update_p2p_metrics(self):
        """Check P2P connectivity between all workers"""
        # Get a random worker to check P2P latency for all other workers
        if len(self.workers) <= 1:
            return
            
        for source_worker in self.workers:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(
                    hostname=source_worker,
                    username=self.ssh_username,
                    key_filename=self.ssh_key_path,
                    timeout=5
                )
                
                for target_worker in self.workers:
                    if source_worker != target_worker:
                        # Run ping from source to target
                        stdin, stdout, stderr = client.exec_command(f"ping -c 3 -W 1 {target_worker}")
                        output = stdout.read().decode()
                        
                        # Parse ping output
                        if 'avg' in output:
                            avg_line = [line for line in output.split('\n') if 'avg' in line][0]
                            avg_str = avg_line.split('=')[1].strip().split('/')[1]
                            latency = float(avg_str)
                            self.metrics[source_worker].p2p_latencies[target_worker] = latency
                            logger.debug(f"P2P latency from {source_worker} to {target_worker}: {latency}ms")
                
                client.close()
                
            except Exception as e:
                logger.error(f"Error checking P2P from {source_worker}: {e}")

    def get_best_devices(self, n):
        """Get n best devices based on current metrics"""
        scores = {}
        for worker, metrics in self.metrics.items():
            if time.time() - metrics.last_update > 30:  # Skip stale data
                continue
            
            # Calculate weighted score (lower is better)
            score = (
                metrics.ping_ms * 0.3 +
                metrics.cpu_percent * 0.4 +
                metrics.memory_percent * 0.3
            )
            scores[worker] = score
        
        sorted_workers = sorted(scores.items(), key=lambda x: x[1])
        return [worker for worker, _ in sorted_workers[:n]]
    
    def get_optimal_chain(self, n):
        """Get optimal chain of n devices based on P2P connectivity and health"""
        if len(self.workers) < n:
            return self.workers
            
        # Start with the healthiest worker
        best_workers = self.get_best_devices(len(self.workers))
        if not best_workers:
            return self.workers[:n]
            
        chain = [best_workers[0]]
        
        # Build chain by finding best next worker at each step
        while len(chain) < n:
            current = chain[-1]
            best_next = None
            best_score = float('inf')
            
            for worker in best_workers:
                if worker in chain:
                    continue
                    
                # Score based on health and P2P latency to current worker
                p2p_latency = self.metrics[current].p2p_latencies.get(worker, 100.0)
                health_score = (
                    self.metrics[worker].ping_ms * 0.2 +
                    self.metrics[worker].cpu_percent * 0.4 +
                    self.metrics[worker].memory_percent * 0.4
                )
                
                # Combined score with higher weight on P2P latency
                score = health_score * 0.6 + p2p_latency * 0.4
                
                if score < best_score:
                    best_score = score
                    best_next = worker
            
            if best_next:
                chain.append(best_next)
            else:
                # If no good next worker, take next best from remaining
                remaining = [w for w in best_workers if w not in chain]
                if remaining:
                    chain.append(remaining[0])
                else:
                    break
        
        return chain[:n]

    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        self.monitor_thread.join()

class ShardWrapper(nn.Module):
    """Wrapper for model shard with P2P support"""
    def __init__(self, submodule, next_worker=None, next_shard_id=None, final=False):
        super().__init__()
        self.module = submodule.to("cpu")
        self.next_worker = next_worker
        self.next_shard_id = next_shard_id
        self.is_final = final
        self.last_inference_time = 0.0
        
        logger.info(f"ShardWrapper initialized with next_worker={next_worker}, next_shard_id={next_shard_id}, is_final={final}")

    def forward(self, x):
        start_time = time.time()
        
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor but got {type(x)}")

        logger.info(f"[{socket.gethostname()}] Received tensor with shape: {x.shape}")

        # Run inference on this shard
        x = x.to("cpu")
        output = self.module(x).cpu()
        
        self.last_inference_time = time.time() - start_time
        logger.info(f"[{socket.gethostname()}] Output shape: {output.shape}, inference time: {self.last_inference_time:.3f}s")
        
        # If this is the final shard, return the output
        if self.is_final:
            logger.info(f"[{socket.gethostname()}] Final shard, returning output")
            return output
            
        # If there's a next worker, forward directly to it
        if self.next_worker and self.next_shard_id is not None:
            try:
                logger.info(f"[{socket.gethostname()}] Forwarding to next worker: {self.next_worker}, shard {self.next_shard_id}")
                # Get worker name for next shard
                worker_name = f"worker{self.next_shard_id}"
                
                # Forward directly to next worker's shard
                return rpc.rpc_sync(self.next_worker, forward_to_worker, args=(output, self.next_shard_id))
            except Exception as e:
                logger.error(f"[{socket.gethostname()}] P2P forwarding failed: {e}")
                # On failure, return the output to the master
                return output
        
        # Default case - return to master
        return output

    def get_inference_time(self):
        return self.last_inference_time
        
    def update_next_worker(self, next_worker, next_shard_id, final=False):
        """Update the next worker in the chain"""
        logger.info(f"Updating next worker to {next_worker}, shard {next_shard_id}, final={final}")
        self.next_worker = next_worker
        self.next_shard_id = next_shard_id
        self.is_final = final
        return True

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

# Helper function for P2P forwarding
def forward_to_worker(x, shard_id):
    """Forward tensor to a specific shard on this worker"""
    worker_name = f"worker{shard_id}"
    try:
        # Get own worker name - this is the worker's rank name (e.g., worker1)
        own_name = rpc.get_worker_info().name
        logger.info(f"[{own_name}] Received P2P forward request for shard {shard_id}")
        
        # Forward to the local shard
        return rpc.rpc_sync(own_name, _local_forward, args=(x, shard_id))
    except Exception as e:
        logger.error(f"Error in forward_to_worker: {e}")
        return x

def _local_forward(x, shard_id):
    """Execute forward pass on a local shard"""
    # This function handles forwarding to a shard on the local worker
    # It's needed because we need to access the local object
    logger.info(f"[{socket.gethostname()}] Local forward to shard {shard_id}")
    
    # The actual implementation will be provided by the runtime system
    # This function assumes worker_shards will be populated at runtime
    # with a dictionary mapping shard_id to ShardWrapper objects
    worker_name = f"worker{shard_id}"
    if not hasattr(_local_forward, "worker_shards"):
        _local_forward.worker_shards = {}
        
    if shard_id in _local_forward.worker_shards:
        shard = _local_forward.worker_shards[shard_id]
        return shard.module(x)
    else:
        logger.error(f"No shard {shard_id} found on this worker")
        return x

class AdaptiveDistributedModel(nn.Module):
    """Distributed model with adaptive shard placement and P2P communication"""
    def __init__(self, model_type: str, num_splits: int, workers: List[str], num_classes: int = 10, 
                 monitor: DeviceMonitor = None):
        super().__init__()
        self.num_splits = num_splits
        self.model_type = model_type
        self.workers = workers
        self.num_classes = num_classes
        self.monitor = monitor
        self.shard_assignments = {}  # {shard_idx: worker_name}
        self.worker_rrefs = {}  # {shard_idx: RRef}
        self.migration_lock = threading.Lock()
        self.batch_count = 0
        self.shard_inference_times = {}  # Track inference times per shard
        self.worker_to_shards = {}  # {worker_name: [shard_indices]}
        
        # Load and initialize model
        model = self._load_model(model_type, num_classes)
        self.shards = split_model_into_n_shards(model, num_splits)
        
        # Initial placement
        self._initial_placement()

    def _load_model(self, model_type, num_classes):
        """Load the model based on type"""
        if model_type == "mobilenetv2":
            model = torchvision_models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mobilenetv2_cifar10.pth"), map_location="cpu"))
        elif model_type == "resnet18":
            model = torchvision_models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet18_cifar10.pth"), map_location="cpu"))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.eval()
        return model

    def _initial_placement(self):
        """Initial placement of shards on workers with P2P chain"""
        if self.monitor:
            # Get optimal chain of workers
            worker_chain = self.monitor.get_optimal_chain(min(self.num_splits, len(self.workers)))
            # Fill with round-robin if needed
            while len(worker_chain) < self.num_splits:
                worker_chain.extend(worker_chain[:self.num_splits - len(worker_chain)])
        else:
            # Default round-robin assignment
            worker_chain = []
            for i in range(self.num_splits):
                worker_chain.append(self.workers[i % len(self.workers)])

        logger.info(f"Initial worker chain: {worker_chain}")
        
        # Deploy shards to workers in the chain
        for i in range(self.num_splits):
            worker_name = worker_chain[i]
            next_worker = worker_chain[(i + 1) % len(worker_chain)] if i < self.num_splits - 1 else None
            next_shard_id = i + 1 if i < self.num_splits - 1 else None
            is_final = (i == self.num_splits - 1)
            
            self._deploy_shard(i, self.shards[i], worker_name, next_worker, next_shard_id, is_final)
            
            # Track which shards are on which worker
            if worker_name not in self.worker_to_shards:
                self.worker_to_shards[worker_name] = []
            self.worker_to_shards[worker_name].append(i)

    def _deploy_shard(self, shard_idx, shard, worker_name, next_worker=None, next_shard_id=None, is_final=False):
        """Deploy a shard to a specific worker with P2P link"""
        logger.info(f"Deploying shard {shard_idx} to {worker_name} (next: {next_worker}, next_shard_id: {next_shard_id}, final: {is_final})")
        
        # Create the shard wrapper on the remote worker
        rref = rpc.remote(worker_name, ShardWrapper, args=(shard, next_worker, next_shard_id, is_final))
        
        # Store the RRef
        self.worker_rrefs[shard_idx] = rref
        self.shard_assignments[shard_idx] = worker_name

    def forward(self, x):
        """Forward pass with P2P communication and adaptive placement"""
        self.batch_count += 1
        
        # Check if we should adapt shard placement
        if self.batch_count % 5 == 0 and self.monitor:  # Check every 5 batches
            self._adapt_shard_placement()
        
        logger.info(f"Starting inference on batch {self.batch_count}")
        
        # Start the chain with the first shard
        try:
            # Send to first shard and let P2P chain handle the rest
            first_shard_rref = self.worker_rrefs[0]
            output = first_shard_rref.rpc_sync().forward(x)
            
            # Collect inference times
            self._collect_inference_times()
            
            return output
            
        except Exception as e:
            logger.error(f"P2P inference chain failed: {e}")
            # Fall back to sequential processing through master
            return self._fallback_forward(x)

    def _fallback_forward(self, x):
        """Fallback to master-coordinated forward pass"""
        logger.warning("Using fallback master-coordinated inference")
        x_tensor = x
        
        for i in range(self.num_splits):
            shard_rref = self.worker_rrefs[i]
            worker_name = self.shard_assignments[i]
            logger.info(f"Fallback: sending to shard {i} on {worker_name}")
            
            try:
                x_tensor = shard_rref.rpc_sync().forward(x_tensor)
            except Exception as e:
                logger.error(f"Error in fallback for shard {i}: {e}")
                self._emergency_migrate_shard(i)
                x_tensor = self.worker_rrefs[i].rpc_sync().forward(x_tensor)
        
        return x_tensor

    def _collect_inference_times(self):
        """Collect inference times from all shards"""
        for i in range(self.num_splits):
            try:
                shard_rref = self.worker_rrefs[i]
                inference_time = shard_rref.rpc_sync().get_inference_time()
                
                if i not in self.shard_inference_times:
                    self.shard_inference_times[i] = deque(maxlen=5)
                self.shard_inference_times[i].append(inference_time)
                
            except Exception as e:
                logger.error(f"Failed to get inference time for shard {i}: {e}")

    def _adapt_shard_placement(self):
        """Adapt shard placement based on performance metrics"""
        with self.migration_lock:
            if not self.monitor:
                return
            
            # Identify problematic shards
            problematic_shards = []
            
            for shard_idx, times in self.shard_inference_times.items():
                if not times:
                    continue
                    
                avg_time = sum(times) / len(times)
                if avg_time > INFERENCE_TIME_THRESHOLD:
                    problematic_shards.append((shard_idx, avg_time))
            
            if not problematic_shards:
                return
            
            # Sort by slowest first
            problematic_shards.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Problematic shards: {problematic_shards}")
            
            # Get best available devices based on current chain
            best_devices = self.monitor.get_optimal_chain(len(self.workers))
            
            # Migrate problematic shards
            migrated_count = 0
            for shard_idx, _ in problematic_shards:
                if migrated_count >= 2:  # Limit to 2 migrations at a time
                    break
                
                current_worker = self.shard_assignments[shard_idx]
                
                # Find a better worker
                better_worker = self._find_better_worker(shard_idx, current_worker, best_devices)
                
                if better_worker:
                    self._migrate_shard(shard_idx, better_worker)
                    migrated_count += 1
                    
            # Update P2P chain if any migrations occurred
            if migrated_count > 0:
                self._update_p2p_chain()

    def _find_better_worker(self, shard_idx, current_worker, best_devices):
        """Find a better worker for a problematic shard"""
        # Get previous and next shard assignments
        prev_shard_idx = shard_idx - 1 if shard_idx > 0 else None
        next_shard_idx = shard_idx + 1 if shard_idx < self.num_splits - 1 else None
        
        prev_worker = self.shard_assignments.get(prev_shard_idx) if prev_shard_idx is not None else None
        next_worker = self.shard_assignments.get(next_shard_idx) if next_shard_idx is not None else None
        
        # Find workers with good connectivity to prev and next
        candidates = []
        
        for worker in best_devices:
            if worker == current_worker:
                continue
                
            # Skip if worker already has too many shards
            if worker in self.worker_to_shards and len(self.worker_to_shards[worker]) >= 2:
                continue
                
            # Calculate worker score
            worker_metrics = self.monitor.metrics.get(worker)
            if not worker_metrics:
                continue
                
            # Basic health score
            health_score = (
                worker_metrics.ping_ms * 0.2 + 
                worker_metrics.cpu_percent * 0.4 + 
                worker_metrics.memory_percent * 0.4
            )
            
            # P2P score
            p2p_score = 0
            if prev_worker:
                p2p_score += worker_metrics.p2p_latencies.get(prev_worker, 100.0)
            if next_worker:
                p2p_score += worker_metrics.p2p_latencies.get(next_worker, 100.0)
                
            # Combined score
            total_score = health_score * 0.7 + p2p_score * 0.3
            
            candidates.append((worker, total_score))
        
        # Sort by score (lower is better)
        candidates.sort(key=lambda x: x[1])
        
        # Return best candidate if significantly better
        if candidates and candidates[0][1] < self._get_worker_score(current_worker) * 0.8:
            return candidates[0][0]
            
        return None

    def _get_worker_score(self, worker):
        """Calculate a worker's score (lower is better)"""
        if not self.monitor:
            return 0
            
        metrics = self.monitor.metrics.get(worker)
        if not metrics:
            return float('inf')
            
        return (
            metrics.ping_ms * 0.2 + 
            metrics.cpu_percent * 0.4 + 
            metrics.memory_percent * 0.4
        )

    def _migrate_shard(self, shard_idx, new_worker):
        """Migrate a shard to a new worker"""
        logger.info(f"Migrating shard {shard_idx} from {self.shard_assignments[shard_idx]} to {new_worker}")
        
        try:
            old_worker = self.shard_assignments[shard_idx]
            
            # Deploy to new worker
            self._deploy_shard(shard_idx, self.shards[shard_idx], new_worker)
            
            # Update worker_to_shards
            if old_worker in self.worker_to_shards:
                self.worker_to_shards[old_worker].remove(shard_idx)
            
            if new_worker not in self.worker_to_shards:
                self.worker_to_shards[new_worker] = []
            self.worker_to_shards[new_worker].append(shard_idx)
            
            logger.info(f"Successfully migrated shard {shard_idx} to {new_worker}")
            
        except Exception as e:
            logger.error(f"Failed to migrate shard {shard_idx}: {e}")

    def _update_p2p_chain(self):
        """Update the P2P chain after migration"""
        logger.info("Updating P2P chain after migration")
        
        # Update each shard's next_worker
        for i in range(self.num_splits):
            next_shard_idx = i + 1 if i < self.num_splits - 1 else None
            next_worker = self.shard_assignments.get(next_shard_idx) if next_shard_idx is not None else None
            is_final = (i == self.num_splits - 1)
            
            try:
                shard_rref = self.worker_rrefs[i]
                shard_rref.rpc_sync().update_next_worker(next_worker, next_shard_idx, is_final)
            except Exception as e:
                logger.error(f"Failed to update P2P link for shard {i}: {e}")

    def _emergency_migrate_shard(self, shard_idx):
        """Emergency migration when a shard fails"""
        logger.warning(f"Emergency migration for shard {shard_idx}")
        
        current_worker = self.shard_assignments[shard_idx]
        available_workers = [w for w in self.workers if w != current_worker]
        
        if available_workers:
            new_worker = random.choice(available_workers)
            self._migrate_shard(shard_idx, new_worker)
            self._update_p2p_chain()
        else:
            logger.error(f"No alternative workers available for shard {shard_idx}")

def split_model_into_n_shards(model: nn.Module, n: int) -> List[nn.Sequential]:
    """Split model into n shards (same as original implementation)"""
    if isinstance(model, torchvision_models.MobileNetV2):
        # MobileNetV2: split features, last shard does pool + flatten + classifier
        feature_layers = list(model.features.children())
        num_feature_layers = len(feature_layers)
        split_idx = num_feature_layers // (n - 1)

        shards = []
        for i in range(n - 1):
            start = i * split_idx
            end = (i + 1) * split_idx if i < n - 2 else num_feature_layers
            shards.append(nn.Sequential(*feature_layers[start:end]))

        final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            *list(model.classifier.children())
        )
        shards.append(final_block)
        return shards

    elif isinstance(model, torchvision_models.ResNet):
        # ResNet18: use layer1 to layer4 blocks, then avgpool + fc
        body = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        body_layers = list(body.children())
        split_idx = len(body_layers) // (n - 1)

        shards = []
        for i in range(n - 1):
            start = i * split_idx
            end = (i + 1) * split_idx if i < n - 2 else len(body_layers)
            shards.append(nn.Sequential(*body_layers[start:end]))

        final_block = nn.Sequential(
            model.avgpool,
            nn.Flatten(),
            model.fc
        )
        shards.append(final_block)
        return shards

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

def run_adaptive_inference(rank, world_size, args):
    """Run adaptive distributed inference with P2P communication"""
    
    # Set up logging
    hostname = socket.gethostname()
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.hostname = hostname
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    # Load environment variables
    load_dotenv()
    master_addr = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '29555')
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    if rank == 0:  # Master node
        os.environ['GLOO_SOCKET_IFNAME'] = MASTER_INTERFACE
    else:  # Worker nodes
        os.environ['GLOO_SOCKET_IFNAME'] = WORKER_INTERFACE
    
    # Initialize RPC
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=4,
        rpc_timeout=3600,
        _transports=["uv"]
    )
    
    if rank == 0:  # Master node
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        
        # Define worker names
        workers = [f"worker{i}" for i in range(1, world_size)]
        
        # Create device monitor
        monitor = DeviceMonitor(
            workers=workers,
            ssh_username=SSH_USERNAME,
            ssh_key_path=SSH_KEY_PATH
        )
        
        # Wait for initial metrics collection
        logger.info("Collecting initial device metrics...")
        time.sleep(10)
        
        # Create adaptive distributed model
        model = AdaptiveDistributedModel(
            model_type=args.model,
            num_splits=args.num_partitions,
            workers=workers,
            num_classes=args.num_classes,
            monitor=monitor
        )
        
        # Load dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset_path = os.path.expanduser('~/datasets/cifar10')
        test_dataset = datasets.CIFAR10(
            root=dataset_path,
            train=False,
            download=False,
            transform=transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True
        )
        
        # Run inference
        logger.info("Starting adaptive P2P inference...")
        start_time = time.time()
        
        total_images = 0
        num_correct = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if i == args.num_batches:
                    break
                
                logger.info(f"Processing batch {i+1}/{args.num_batches}")
                batch_start_time = time.time()
                
                output = model(images)
                
                batch_time = time.time() - batch_start_time
                logger.info(f"Batch {i} completed in {batch_time:.2f}s")
                
                _, predicted = torch.max(output.data, 1)
                num_correct += (predicted == labels).sum().item()
                total_images += len(images)
        
        elapsed_time = time.time() - start_time
        accuracy = 100.0 * num_correct / total_images
        
        logger.info(f"Inference completed in {elapsed_time:.2f}s")
        logger.info(f"Total images: {total_images}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        
        # Cleanup
        monitor.stop()
        
    else:  # Worker nodes
        worker_name = f"worker{rank}"
        rpc.init_rpc(
            worker_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        logger.info(f"Worker {rank} initialized and ready for P2P communication")
    
    # Wait for RPC shutdown
    rpc.shutdown()

def main():
    parser = argparse.ArgumentParser(description="P2P Adaptive Distributed DNN Inference")
    parser.add_argument("--rank", type=int, default=0, help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, help="World size")
    parser.add_argument("--model", type=str, default="mobilenetv2", 
                        choices=["mobilenetv2", "resnet18"],
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to run")
    parser.add_argument("--num-partitions", type=int, default=2, help="Number of model partitions")
    
    args = parser.parse_args()
    
    run_adaptive_inference(
        rank=args.rank,
        world_size=args.world_size,
        args=args
    )

if __name__ == "__main__":
    main()