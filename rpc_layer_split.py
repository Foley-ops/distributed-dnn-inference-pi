#!/usr/bin/env python3

import os
import time
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.optim as optim
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dotenv import load_dotenv
import argparse
from typing import List, Tuple, Dict, Optional

# Import available lightweight models
from models.mobilenetv2.model import MobileNetV2
from models.squeezenet.model import SqueezeNet
from models.efficientnet_b0.model import EfficientNetB0

# Base class for model shards
class ModelShardBase(nn.Module):
    def __init__(self, device):
        super(ModelShardBase, self).__init__()
        self.device = device
    
    def forward(self, x_rref):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def parameter_rrefs(self):
        """
        Returns a list of RRefs to model parameters for distributed optimizer
        """
        param_rrefs = []
        for param in self.parameters():
            param_rrefs.append(RRef(param))
        return param_rrefs

# First half of MobileNetV2
class MobileNetV2Shard1(ModelShardBase):
    def __init__(self, device, num_classes=1000):
        super(MobileNetV2Shard1, self).__init__(device)
        
        # Create complete model then extract first half
        #TODO add changes for pretrained models on pytorch library
        complete_model = MobileNetV2(num_classes=num_classes)
        
        # First shard includes the features up to halfway point
        features = complete_model.features
        split_idx = len(features) // 2
        
        self.features_first_half = nn.Sequential(*list(features.children())[:split_idx])
        self.features_first_half.to(self.device)
    
    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        output = self.features_first_half(x)
        # Return to CPU for RPC transfer
        return output.cpu()

# Second half of MobileNetV2
class MobileNetV2Shard2(ModelShardBase):
    def __init__(self, device, num_classes=1000):
        super(MobileNetV2Shard2, self).__init__(device)
        
        # Create complete model then extract second half
        complete_model = MobileNetV2(num_classes=num_classes)
        
        # Extract the second half of features and the classifier
        features = complete_model.features
        split_idx = len(features) // 2
        
        self.features_second_half = nn.Sequential(*list(features.children())[split_idx:])
        self.classifier = complete_model.classifier
        
        self.features_second_half.to(self.device)
        self.classifier.to(self.device)
    
    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        x = self.features_second_half(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # Return to CPU for RPC transfer
        return x.cpu()

# Similar pattern for SqueezeNet and EfficientNetB0 shards...
# (Implementations omitted for brevity, but would follow same pattern)

# Distributed model using pipeline parallelism
class DistributedModel(nn.Module):
    def __init__(self, model_type: str, num_splits: int, workers: List[str], num_classes: int = 1000):
        super(DistributedModel, self).__init__()
        self.model_type = model_type
        self.num_splits = num_splits
        
        # Map model_type to appropriate shard classes
        shard_classes = {
            'mobilenetv2': (MobileNetV2Shard1, MobileNetV2Shard2),
            'squeezenet': (None, None),  # Replace with actual implementations
            'efficientnet_b0': (None, None)  # Replace with actual implementations
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
        for x in iter(xs.split(self.num_splits, dim=0)):
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
        """
        Returns all parameter RRefs from both model shards for distributed optimizer
        """
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params

def run_inference(rank, world_size, model_type, batch_size, num_micro_batches, num_classes, dataset):
    """
    Main function to run distributed inference
    """
    # Get master IP from .env file
    master_ip = os.getenv('MASTER_IP', 'localhost')
    master_port = os.getenv('MASTER_PORT', '55555')
    
    # Initialize RPC framework
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    
    # Define RPC names for workers
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=4, #num threads per worker
        rpc_timeout=0  # infinity
    )
    
    if rank == 0:  # Master LAMBDA
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        
        # Define worker names
        workers = [f"worker{i}" for i in range(1, world_size)]
        
        # Create distributed model
        model = DistributedModel(
            model_type=model_type,
            num_splits=num_micro_batches,
            workers=workers,
            num_classes=num_classes
        )
        
        # Load CIFAR-10 data if specified
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((224, 224))  # Resize to match model input size
            ])
            
            test_dataset = datasets.CIFAR10(
                root='~/datasets/cifar10',  # Using your existing dataset path
                train=False, 
                download=False,  # Dataset already exists
                transform=transform
            )
            
            # Select a batch for testing
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True
            )
            
            # Get a batch of images
            images, _ = next(iter(test_loader))
        else:
            # Fall back to dummy data if dataset not specified
            images = torch.randn(batch_size, 3, 224, 224)
        
        # Time the inference
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        elapsed_time = time.time() - start_time
        
        print(f"Inference time: {elapsed_time:.4f} seconds")
        print(f"Output shape: {output.shape}")
        
    else:  # Workers
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        
        # Workers just wait for RPCs from the master
    
    # Block until all RPCs finish
    rpc.shutdown()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Distributed DNN Inference on Raspberry Pi")
    parser.add_argument("--rank", type=int, default=0, help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, help="World size (1 master + N workers)")
    parser.add_argument("--model", type=str, default="mobilenetv2", 
                        choices=["mobilenetv2", "squeezenet", "efficientnet_b0"], 
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--micro-batches", type=int, default=4, help="Number of micro-batches for pipeline")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes") # 10 for CIFAR-10
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["cifar10", "mnist"],
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