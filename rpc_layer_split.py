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
from datetime import timedelta
import argparse
import logging
import socket
import sys
from typing import List
import psutil 


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
        # complete_model = torchvision_models.mobilenet_v2(num_classes=num_classes)
        # complete_model = torchvision_models.mobilenet_v2(pretrained=True)
        complete_model = torchvision_models.mobilenet_v2(weights=None)  

        # load the saved state dictionary, ensuring it is mapped to the correct device
        state_dict = torch.load("mobilenetv2_cifar10.pth", map_location=torch.device(device)) # model path is hard coded for now 

        # adjust number of output classes if needed 
        if num_classes != 1000: 
            complete_model.classifier[1] = nn.Linear(complete_model.last_channel, num_classes)

        # load the saved weights into complete_model    
        complete_model.load_state_dict(state_dict)
        complete_model.eval()

        # First shard includes the features up to halfway point
        features = complete_model.features
        split_idx = len(features) // 2
        
        self.features_first_half = nn.Sequential(*list(features.children())[:split_idx])
        self.features_first_half.to(self.device)
    
    def forward(self, x_rref):
        logging.info(f"MobileNetV2Shard1: Received input tensor with shape {x_rref.to_here().shape}")

        x = x_rref.to_here().to(self.device)

        worker_inference_start_time = time.time() # record start time
        cpu_percent = psutil.cpu_percent(0.1)

        output = self.features_first_half(x) # inference 
        
        # temporary memory stats collection 
        memory = psutil.virtual_memory()
        memory_used_gb = round((memory.total - memory.available) / (1024**3), 2)
        memory_percent = memory.percent
        logging.info(f"Memory Usage Percent: {memory_percent}")
        logging.info(f"Memory Used GB: {memory_used_gb}")

        # temporary CPU usge percent collection 
        cpu_percent = psutil.cpu_percent(0.1)
        logging.info(f"CPU Usage Percentage (%): {cpu_percent}")

        worker_inference_total_time = time.time() - worker_inference_start_time # calculate time spent on inference 
        logging.info(f"Time spent on inference: {worker_inference_total_time}")

        logging.info(f"MobileNetV2Shard1: Produced output tensor with shape {output.shape}")
        # Return to CPU for RPC transfer
        return output.cpu()

# Second half of MobileNetV2
class MobileNetV2Shard2(ModelShardBase):
    def __init__(self, device, num_classes=10):
        super(MobileNetV2Shard2, self).__init__(device)
        
        # Use torchvision's MobileNetV2
        # complete_model = torchvision_models.mobilenet_v2(num_classes=num_classes)
        complete_model = torchvision_models.mobilenet_v2(weights=None)
        state_dict = torch.load("mobilenetv2_cifar10.pth", map_location=torch.device(device))

        # adjust number of output classes if needed 
        if num_classes != 1000: 
            complete_model.classifier[1] = nn.Linear(complete_model.last_channel, num_classes)

        # load the saved weights into complete_model
        complete_model.load_state_dict(state_dict)
        complete_model.eval()

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

        worker_inference_start_time = time.time() # record start time
        cpu_percent = psutil.cpu_percent(0.1)

        x = self.features_second_half(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # temporary memory stats collection 
        memory = psutil.virtual_memory()
        memory_used_gb = round((memory.total - memory.available) / (1024**3), 2)
        memory_percent = memory.percent
        logging.info(f"Memory Usage Percent: {memory_percent}")
        logging.info(f"Memory Used GB: {memory_used_gb}")

        # temporary CPU usge percent collection 
        cpu_percent = psutil.cpu_percent(0.1)
        logging.info(f"CPU Usage Percentage (%): {cpu_percent}")

        worker_inference_total_time = time.time() - worker_inference_start_time # calculate time spent on inference 
        logging.info(f"Time spent on inference: {worker_inference_total_time}")

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

def run_inference(rank, world_size, model_type, batch_size, num_micro_batches, num_classes, dataset, num_batches):
    """
    Main function to run distributed inference
    """

    connected = True

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
    
    # Get master address from .env file
    master_addr = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '29555')
    
    logger.info(f"Using master address: {master_addr} and port: {master_port}")
    
    # Initialize RPC framework
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    if rank == 0:  # Only on master node
        # Specify the interface with the master node IP address
        os.environ['GLOO_SOCKET_IFNAME'] = 'enp6s0'  # Using your wired interface
        os.environ['TENSORPIPE_SOCKET_IFADDR'] = '0.0.0.0'
        logger.info(f"Set GLOO_SOCKET_IFNAME to enp6s0 for binding")
    else:  # Only on worker nodes
        # For WiFi connections on Raspberry Pis
        os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'  # Typical WiFi interface name on Pis
        logger.info(f"Set GLOO_SOCKET_IFNAME to bind to WiFi interface")
    
    # Flag to track if RPC was successfully initialized
    rpc_initialized = False
    
    if rank == 0:  # Master node
        logger.info("Initializing master node")
        try:
            # Create a more explicit RPC backend options
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=4,
                rpc_timeout=3600,
                _transports=["uv"],  # Force using UV transport only
                init_method=f"tcp://0.0.0.0:{master_port}"  # Explicit init method
            )
            
            logger.info(f"Master using explicit init_method: tcp://0.0.0.0:{master_port}")
            
            # Initialize RPC for master
            rpc.init_rpc(
                "master",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options
            )
            logger.info("Master RPC initialized successfully")
            rpc_initialized = True
            
            # Define worker names
            workers = [f"worker{i}" for i in range(1, world_size)]
            logger.info(f"Setting up model with workers: {workers}")
            
            # Create distributed model
            model = DistributedModel(
                model_type=model_type,
                num_splits=num_micro_batches,
                workers=workers,
                num_classes=num_classes
            )
            logger.info("Distributed model created successfully")
            
            # Load data
            logger.info(f"Loading {dataset} dataset")
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
                
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=True
                )
                
                images, labels = next(iter(test_loader))
                logger.info(f"Loaded batch of {len(images)} images with shape: {images.shape}")
                logger.info(f"First few labels: {labels[:5]}")
            else:
                images = torch.randn(batch_size, 3, 224, 224)
                logger.info(f"Using dummy data with shape: {images.shape}")
            
            # Run inference
            logger.info("Starting inference...")
            start_time = time.time()
            
            total_images = 0
            num_correct = 0

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    if i == num_batches:
                        break
                        
                    logger.info(f"Running inference on batch {i+1}/{num_batches} with shape: {images.shape}")
                    batch_start_time = time.time()
                    output = model(images)
                    batch_total_time = time.time() - batch_start_time
                    logger.info(f"End to end time for batch {i}: {batch_total_time}")

                    logger.info(f"Received output shape: {output.shape}")

                    # log the predicted vs actual labels
                    _, predicted = torch.max(output.data, 1)
                    logger.info(f"Predicted: {predicted[:5]} | Actual: {labels[:5]}")

                    if (predicted[:5] == labels[:5]):
                        num_correct += len(images) 

                    total_images += len(images)

            elapsed_time = time.time() - start_time
            logger.info(f"Inference completed on {total_images} images.")
            logger.info(f"Final Accuracy: {1.0 * num_correct / total_images * 100.0}") # only works with batch_size=1 for now
            
            logger.info(f"Inference completed in {elapsed_time:.4f} seconds")
            
            # Print some results
            if dataset == 'cifar10':
                _, predicted = torch.max(output.data, 1)
                logger.info(f"First few predictions: {predicted[:5]}")
                logger.info(f"First few actual labels: {labels[:5]}")
            
        except Exception as e:
            logger.error(f"Error in master node: {str(e)}", exc_info=True)
            
    else:  # Workers
        logger.info(f"Initializing worker node with rank {rank}")
        retry_count = 0
        max_retries = 30  # More retries
        connected = False
        
        while retry_count < max_retries and not connected:
            try:
                # Force workers to use WiFi interface
                os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'
                logger.info(f"Worker binding to interface: {os.environ.get('GLOO_SOCKET_IFNAME')}")
                
                # Create a more explicit RPC backend options
                rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                    num_worker_threads=4,
                    rpc_timeout=3600,
                    _transports=["uv"],  # Force using UV transport only
                    init_method=f"tcp://{master_addr}:{master_port}"  # Explicit init method
                )
                
                logger.info(f"Worker using explicit init_method: tcp://{master_addr}:{master_port}")
                
                logger.info(f"Worker {rank} attempt {retry_count+1} to connect to master...")
                
                # Check if RPC is already initialized; if so, skip reinitialization
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
                    break  # Exit the loop instead of raising
                wait_time = 10 + (retry_count % 5)
                logger.info(f"Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                time.sleep(wait_time)

    if not connected and rank != 0:
        logger.error("Worker failed to connect to master node")
        # Exit with a non-zero code so the shell script knows to retry
        sys.exit(1)
    
    # Only call shutdown if RPC was successfully initialized
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
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to run during inference")

    
    args = parser.parse_args()
    
    # Run inference
    run_inference(
        rank=args.rank,
        world_size=args.world_size,
        model_type=args.model,
        batch_size=args.batch_size,
        num_micro_batches=args.micro_batches,
        num_classes=args.num_classes,
        dataset=args.dataset,
        num_batches=args.num_batches
    )

if __name__ == "__main__":
    main()