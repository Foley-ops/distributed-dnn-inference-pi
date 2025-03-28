#!/usr/bin/env python3

# Disable PyTorch's advanced CPU optimizations for Raspberry Pi compatibility
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
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s [%(hostname)s:rank%(rank)s]',
)

# Base class for model shards
class ModelShardBase(nn.Module):
    def __init__(self, device):
        super(ModelShardBase, self).__init__()
        self.device = device
    
    def forward(self, x_rref):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def parameter_rrefs(self):
        param_rrefs = []
        for param in self.parameters():
            param_rrefs.append(RRef(param))
        return param_rrefs

# First half of MobileNetV2
class MobileNetV2Shard1(ModelShardBase):
    def __init__(self, device, num_classes=10):
        super(MobileNetV2Shard1, self).__init__(device)
        
        # Use torchvision's MobileNetV2
        complete_model = torchvision_models.mobilenet_v2(num_classes=num_classes)
        
        # First shard includes the features up to halfway point
        features = complete_model.features
        split_idx = len(features) // 2
        
        self.features_first_half = nn.Sequential(*list(features.children())[:split_idx])
        self.features_first_half.to(self.device)
    
    def forward(self, x_rref):
        logging.info(f"MobileNetV2Shard1: Received input tensor with shape {x_rref.to_here().shape}")
        x = x_rref.to_here().to(self.device)
        output = self.features_first_half(x)
        logging.info(f"MobileNetV2Shard1: Produced output tensor with shape {output.shape}")
        # Return to CPU for RPC transfer
        return output.cpu()

# Second half of MobileNetV2
class MobileNetV2Shard2(ModelShardBase):
    def __init__(self, device, num_classes=10):
        super(MobileNetV2Shard2, self).__init__(device)
        
        # Use torchvision's MobileNetV2
        complete_model = torchvision_models.mobilenet_v2(num_classes=num_classes)
        
        # Extract the second half of features and the classifier
        features = complete_model.features
        split_idx = len(features) // 2
        
        self.features_second_half = nn.Sequential(*list(features.children())[split_idx:])
        self.classifier = complete_model.classifier
        
        self.features_second_half.to(self.device)
        self.classifier.to(self.device)
    
    def forward(self, x_rref):
        logging.info(f"MobileNetV2Shard2: Received input tensor with shape {x_rref.to_here().shape}")
        x = x_rref.to_here().to(self.device)
        x = self.features_second_half(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        logging.info(f"MobileNetV2Shard2: Produced output tensor with shape {x.shape}")
        # Return to CPU for RPC transfer
        return x.cpu()

# Distributed model using pipeline parallelism
class DistributedModel(nn.Module):
    def __init__(self, model_type: str, num_splits: int, workers: List[str], num_classes: int = 10):
        super(DistributedModel, self).__init__()
        self.model_type = model_type
        self.num_splits = num_splits
        
        # Map model_type to appropriate shard classes
        shard_classes = {
            'mobilenetv2': (MobileNetV2Shard1, MobileNetV2Shard2),
            # Other models can be added here
        }
        
        if model_type not in shard_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        Shard1, Shard2 = shard_classes[model_type]
        
        # Put the first part of the model on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            Shard1,
            args=("cpu", num_classes)
        )
        
        # Put the second part of the model on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            Shard2,
            args=("cpu", num_classes)
        )
    
    def forward(self, xs):
        # Pipeline parallelism implementation
        out_futures = []
        
        # Split input batch into micro-batches
        for i, x in enumerate(iter(xs.split(self.num_splits, dim=0))):
            logging.info(f"Processing micro-batch {i+1}/{self.num_splits}")
            # Create RRef for the input data
            x_rref = RRef(x)
            
            # Forward through first shard
            y_rref = self.p1_rref.remote().forward(x_rref)
            
            # Forward through second shard (asynchronously)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)
        
        # Collect and concatenate all outputs
        return torch.cat(torch.futures.wait_all(out_futures))
    
    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params

def run_inference(rank, world_size, model_type, batch_size, num_micro_batches, num_classes, dataset):
    """
    Main function to run distributed inference
    """
    # Add hostname to log formatter
    hostname = socket.gethostname()
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.hostname = hostname
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting distributed inference process")

    # Load environment variables
    load_dotenv()
    
    # Get master address and port from .env file
    master_addr = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '29555')
    logger.info(f"Using master address: {master_addr} and port: {master_port}")
    
    # Set environment variables for RPC
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Set network interface for GLOO and TensorPipe
    if rank == 0:  # Master node
        os.environ['GLOO_SOCKET_IFNAME'] = 'enp6s0'
        os.environ['TP_SOCKET_IFNAME'] = 'enp6s0'
        logger.info("Set GLOO_SOCKET_IFNAME and TP_SOCKET_IFNAME to enp6s0 for master")
    else:  # Worker nodes
        os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'
        os.environ['TP_SOCKET_IFNAME'] = 'wlan0'
        logger.info("Set GLOO_SOCKET_IFNAME and TP_SOCKET_IFNAME to wlan0 for worker")
    
    # Define RPC backend options with _transports to disable shared-memory transport
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=4,
        rpc_timeout=120,  # 2 minute timeout
        _transports=["uv"]
    )
    
    # Flag to track if RPC was successfully initialized
    rpc_initialized = False
    
    if rank == 0:  # Master node
        logger.info("Initializing master node")
        try:
            rpc.init_rpc(
                "master",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options
            )
            logger.info("Master RPC initialized successfully")
            rpc_initialized = True
            # Rest of master code...
        except Exception as e:
            logger.error(f"Error in master node: {str(e)}", exc_info=True)
    else:  # Worker nodes
        logger.info(f"Initializing worker node with rank {rank}")
        retry_count = 0
        max_retries = 30
        connected = False
        
        while retry_count < max_retries and not connected:
            try:
                # Force workers to use WiFi interface
                os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'
                os.environ['TP_SOCKET_IFNAME'] = 'wlan0'
                logger.info(f"Worker binding to interface: {os.environ.get('GLOO_SOCKET_IFNAME')}")
                logger.info(f"Worker {rank} attempt {retry_count+1} to connect to master...")
                
                if hasattr(rpc, 'is_initialized') and rpc.is_initialized():
                    logger.info("RPC is already initialized; skipping reinitialization.")
                    connected = True
                    rpc_initialized = True
                    break
                
                rpc.init_rpc(
                    f"worker{rank}",
                    rank=rank,
                    world_size=world_size,
                    rpc_backend_options=rpc_backend_options
                )
                logger.info(f"Worker {rank} RPC initialized successfully")
                connected = True
                rpc_initialized = True
            except Exception as e:
                retry_count += 1
                logger.warning(f"Connection attempt {retry_count} failed: {str(e)}")
                if retry_count >= max_retries:
                    logger.error(f"Worker {rank} failed to connect after {max_retries} attempts")
                    break
                wait_time = 10 + (retry_count % 5)
                logger.info(f"Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                time.sleep(wait_time)
    
    if rpc_initialized:
        logger.info("Waiting for RPC shutdown")
        try:
            rpc.shutdown()
            logger.info("RPC shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    else:
        logger.warning("RPC was never successfully initialized, skipping shutdown")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Distributed DNN Inference on Raspberry Pi")
    parser.add_argument("--rank", type=int, default=0, help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, help="World size (1 master + N workers)")
    parser.add_argument("--model", type=str, default="mobilenetv2", 
                        choices=["mobilenetv2"],
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--micro-batches", type=int, default=2, help="Number of micro-batches for pipeline")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["cifar10", "dummy"],
                        help="Dataset to use for inference")
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(
        rank=args.rank,
        world_size=args.world_size,
        model_type=args.model,
        batch_size=args.batch_size,
        num_micro_batches=args.micro_batches,
        num_classes=args.num_classes,
        dataset=args.dataset
    )

if __name__ == "__main__":
    main()