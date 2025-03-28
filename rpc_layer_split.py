#!/usr/bin/env python3
import os
import time
import socket
import sys
import argparse
import logging

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, TensorPipeRpcBackendOptions
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
from dotenv import load_dotenv
from typing import List

# -------------------------------------------------------------------
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
        complete_model = torchvision_models.mobilenet_v2(num_classes=num_classes)
        features = complete_model.features
        split_idx = len(features) // 2
        self.features_first_half = nn.Sequential(*list(features.children())[:split_idx])
        self.features_first_half.to(self.device)

    def forward(self, x_rref):
        logging.info(f"MobileNetV2Shard1: Received input tensor with shape {x_rref.to_here().shape}")
        x = x_rref.to_here().to(self.device)
        logging.info(f"MobileNetV2Shard1: Starting computation...")
        output = self.features_first_half(x)
        logging.info(f"MobileNetV2Shard1: Computation complete")
        logging.info(f"MobileNetV2Shard1: Produced output tensor with shape {output.shape}")
        return output.cpu()

# Second half of MobileNetV2
class MobileNetV2Shard2(ModelShardBase):
    def __init__(self, device, num_classes=10):
        super(MobileNetV2Shard2, self).__init__(device)
        complete_model = torchvision_models.mobilenet_v2(num_classes=num_classes)
        features = complete_model.features
        split_idx = len(features) // 2
        self.features_second_half = nn.Sequential(*list(features.children())[split_idx:])
        self.classifier = complete_model.classifier
        self.features_second_half.to(self.device)
        self.classifier.to(self.device)

    def forward(self, x_rref):
        logging.info(f"MobileNetV2Shard2: Received input tensor with shape {x_rref.to_here().shape}")
        x = x_rref.to_here().to(self.device)
        logging.info(f"MobileNetV2Shard2: Starting computation...")
        x = self.features_second_half(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        logging.info(f"MobileNetV2Shard2: Computation complete")
        logging.info(f"MobileNetV2Shard2: Produced output tensor with shape {x.shape}")
        return x.cpu()

# Distributed model using pipeline parallelism
class DistributedModel(nn.Module):
    def __init__(self, model_type: str, num_splits: int, workers: List[str], num_classes: int = 10):
        super(DistributedModel, self).__init__()
        self.model_type = model_type
        self.num_splits = num_splits

        shard_classes = {
            'mobilenetv2': (MobileNetV2Shard1, MobileNetV2Shard2),
        }
        if model_type not in shard_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
        Shard1, Shard2 = shard_classes[model_type]

        # Place shards on remote workers.
        self.p1_rref = rpc.remote(
            workers[0],
            Shard1,
            args=("cpu", num_classes)
        )
        self.p2_rref = rpc.remote(
            workers[1],
            Shard2,
            args=("cpu", num_classes)
        )

    def forward(self, xs):
        out_futures = []
        for i, x in enumerate(iter(xs.split(self.num_splits, dim=0))):
            logging.info(f"Processing micro-batch {i+1}/{self.num_splits}")
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params

# -------------------------------------------------------------------
# Robust RPC initialization with retries and proper cleanup.
def init_rpc_with_retries(name, rank, world_size, options, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            rpc.init_rpc(name=name, rank=rank, world_size=world_size, rpc_backend_options=options)
            logging.info(f"[Rank {rank}] RPC initialized successfully on attempt {attempt}.")
            return True
        except Exception as e:
            logging.warning(f"[Rank {rank}] rpc.init_rpc failed on attempt {attempt}: {e}")
            try:
                rpc.shutdown(graceful=False)
            except Exception as cleanup_err:
                logging.warning(f"[Rank {rank}] RPC shutdown during cleanup raised: {cleanup_err}")
            time.sleep(5)
    return False

def run_inference(rank, world_size, model_type, batch_size, num_micro_batches, num_classes, dataset):
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
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get master address and port from .env file
    master_addr = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '29555')
    logger.info(f"Using master address: {master_addr} and port: {master_port}")

    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Configure network interfaces for proper binding
    if rank == 0:
        os.environ['GLOO_SOCKET_IFNAME'] = 'enp6s0'
        os.environ['TP_SOCKET_IFNAME'] = 'enp6s0'
        logger.info("Master using interface enp6s0 for binding")
    else:
        os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'
        os.environ['TP_SOCKET_IFNAME'] = 'wlan0'
        logger.info("Worker binding to interface wlan0")

    # Configure RPC options
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=4,
        rpc_timeout=300  # Increased timeout to 5 minutes
    )
    
    # Initialize RPC with retries
    rpc_initialized = init_rpc_with_retries(
        name="master" if rank == 0 else f"worker{rank}",
        rank=rank,
        world_size=world_size,
        options=options,
        max_retries=3
    )
    
    if not rpc_initialized:
        logger.error(f"[Rank {rank}] RPC initialization failed after multiple attempts")
        sys.exit(1)

    # Rest of the function remains the same
    if rank == 0:  # Master node
        logger.info("Initializing master node")
        workers = [f"worker{i}" for i in range(1, world_size)]
        logger.info(f"Setting up model with workers: {workers}")
        try:
            model = DistributedModel(
                model_type=model_type,
                num_splits=num_micro_batches,
                workers=workers,
                num_classes=num_classes
            )
            logger.info("Distributed model created successfully")
            logger.info(f"Loading {dataset} dataset")
            
            # Load only one image
            if dataset == 'cifar10':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.Resize((224, 224))
                ])
                
                dataset_path = os.path.expanduser('~/datasets/cifar10')
                logger.info(f"Loading CIFAR-10 from: {dataset_path}")
                
                test_dataset = datasets.CIFAR10(
                    root=dataset_path,
                    train=False, 
                    download=False,
                    transform=transform
                )
                
                # Get just a single image and reshape it to add batch dimension
                single_image, single_label = test_dataset[0]  # Get the first image
                single_image = single_image.unsqueeze(0)  # Add batch dimension
                single_label = torch.tensor([single_label])  # Convert to tensor with one element
                
                logger.info(f"Loaded single image with shape: {single_image.shape}")
                logger.info(f"Label: {single_label.item()}")
            else:
                # Generate a single random image
                single_image = torch.randn(1, 3, 224, 224)  # Batch size of 1
                single_label = None
                logger.info(f"Using single dummy image with shape: {single_image.shape}")
            
            logger.info("Starting inference...")
            start_time = time.time()
            with torch.no_grad():
                logger.info("Sending data through the pipeline...")
                output = model(single_image)
                logger.info(f"Received output from pipeline with shape: {output.shape}")
            elapsed_time = time.time() - start_time
            
            logger.info(f"Inference completed in {elapsed_time:.4f} seconds")
            
            if dataset == 'cifar10' and single_label is not None:
                _, predicted = torch.max(output.data, 1)
                logger.info(f"Prediction: {predicted.item()}")
                logger.info(f"Actual label: {single_label.item()}")
            
        except Exception as e:
            logger.error(f"Error in master node: {str(e)}", exc_info=True)
    else:
        logger.info(f"Worker node {rank} is ready and waiting for tasks.")
        # Keep worker alive long enough for the master to complete its RPCs.
        time.sleep(60)

    # Shutdown RPC based on our own initialization flag.
    if rpc_initialized:
        if rank == 0:
            logger.info("Master calling rpc.shutdown()")
            rpc.shutdown()
            logger.info("Master RPC shutdown complete")
        else:
            logger.info(f"Worker {rank} calling rpc.shutdown() after delay")
            rpc.shutdown()
            logger.info(f"Worker {rank} RPC shutdown complete")
    else:
        logger.warning("RPC was not successfully initialized, skipping shutdown")

def main():
    parser = argparse.ArgumentParser(description="Distributed DNN Inference on Raspberry Pi")
    parser.add_argument("--rank", type=int, default=0, help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, help="World size (1 master + N workers)")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2"], help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")  # Reduced from 8
    parser.add_argument("--micro-batches", type=int, default=1, help="Number of micro-batches for pipeline")  # Reduced from 2
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "dummy"], help="Dataset to use for inference")
    args = parser.parse_args()
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