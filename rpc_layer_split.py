#!/usr/bin/env python3

# Disable PyTorch's SIMD optimization for Raspberry Pi compatibility
import os
os.environ['ATEN_CPU_CAPABILITY'] = ''

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

# Use only available lightweight models; remove unavailable ones like SqueezeNet.
import torchvision.models as torchvision_models

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
        self.features_first_half = nn.Sequential(*list(features.children())[:split_idx]).to(self.device)
    
    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        output = self.features_first_half(x)
        return output.cpu()

# Second half of MobileNetV2
class MobileNetV2Shard2(ModelShardBase):
    def __init__(self, device, num_classes=10):
        super(MobileNetV2Shard2, self).__init__(device)
        complete_model = torchvision_models.mobilenet_v2(num_classes=num_classes)
        features = complete_model.features
        split_idx = len(features) // 2
        self.features_second_half = nn.Sequential(*list(features.children())[split_idx:]).to(self.device)
        self.classifier = complete_model.classifier.to(self.device)
    
    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        x = self.features_second_half(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.cpu()

# Distributed model using pipeline parallelism
class DistributedModel(nn.Module):
    def __init__(self, model_type: str, num_splits: int, workers: List[str], num_classes: int = 1000):
        super(DistributedModel, self).__init__()
        if model_type != 'mobilenetv2':
            raise ValueError(f"Unsupported model type: {model_type}")
        self.p1_rref = rpc.remote(workers[0], MobileNetV2Shard1, args=("cpu", num_classes))
        self.p2_rref = rpc.remote(workers[1], MobileNetV2Shard2, args=("cpu", num_classes))
        self.num_splits = num_splits
    
    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_splits, dim=0)):
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

def run_inference(rank, world_size, model_type, batch_size, num_micro_batches, num_classes, dataset):
    from dotenv import load_dotenv
    load_dotenv()
    master_ip = os.getenv('MASTER_IP', 'localhost')
    master_port = os.getenv('MASTER_PORT', '55555')
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4, rpc_timeout=0)
    
    if rank == 0:
        rpc.init_rpc("master", rank=rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
        workers = [f"worker{i}" for i in range(1, world_size)]
        model = DistributedModel(model_type=model_type, num_splits=num_micro_batches, workers=workers, num_classes=num_classes)
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((224, 224))
            ])
            test_dataset = datasets.CIFAR10(root='~/datasets/cifar10', train=False, download=False, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            images, _ = next(iter(test_loader))
        else:
            images = torch.randn(batch_size, 3, 224, 224)
        start_time = time.time()
        with torch.no_grad():
            output = model(images)
        elapsed_time = time.time() - start_time
        print(f"Inference time: {elapsed_time:.4f} seconds")
        print(f"Output shape: {output.shape}")
    else:
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
    
    rpc.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Distributed DNN Inference on Raspberry Pi")
    parser.add_argument("--rank", type=int, default=0, help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, help="World size (1 master + N workers)")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2"], help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--micro-batches", type=int, default=4, help="Number of micro-batches for pipeline")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"], help="Dataset to use for inference")
    
    args = parser.parse_args()
    run_inference(rank=args.rank, world_size=args.world_size, model_type=args.model, batch_size=args.batch_size,
                  num_micro_batches=args.micro_batches, num_classes=args.num_classes, dataset=args.dataset)

if __name__ == "__main__":
    main()
