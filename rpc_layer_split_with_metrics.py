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
import csv
from datetime import datetime
import pickle
import struct


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s [%(hostname)s:rank%(rank)s]',
)


# Metrics collection class
class MetricsCollector:
    def __init__(self, rank, output_dir="./metrics"):
        self.rank = rank
        self.hostname = socket.gethostname()
        self.output_dir = output_dir
        self.metrics_data = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV file path
        self.csv_file = os.path.join(output_dir, f"metrics_rank_{rank}_{self.hostname}.csv")
        
        # Initialize CSV file with headers
        self.init_csv()
    
    def init_csv(self):
        headers = [
            'timestamp', 'datetime', 'rank', 'hostname', 'event_type',
            'batch_id', 'shard_id', 'image_id', 'tensor_shape', 
            'inference_time', 'per_image_inference_time', 'rpc_latency', 
            'data_transfer_size_bytes', 'network_throughput_mbps',
            'memory_percent', 'memory_used_gb', 'cpu_percent',
            'accuracy', 'total_images', 'predictions', 'actual_labels'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def collect_inference_metrics(self, event_type, batch_id=None, shard_id=None, 
                                image_id=None, tensor_shape=None, inference_time=None,
                                per_image_inference_time=None, rpc_latency=None,
                                data_transfer_size_bytes=None, network_throughput_mbps=None,
                                accuracy=None, total_images=None, 
                                predictions=None, actual_labels=None):
        
        # Get system metrics
        memory = psutil.virtual_memory()
        memory_used_gb = round((memory.total - memory.available) / (1024**3), 2)
        memory_percent = memory.percent
        cpu_percent = psutil.cpu_percent()
        
        # Create metrics record
        timestamp = time.time()
        datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
        
        metrics_record = {
            'timestamp': timestamp,
            'datetime': datetime_str,
            'rank': self.rank,
            'hostname': self.hostname,
            'event_type': event_type,
            'batch_id': batch_id,
            'shard_id': shard_id,
            'image_id': image_id,
            'tensor_shape': str(tensor_shape) if tensor_shape is not None else None,
            'inference_time': inference_time,
            'per_image_inference_time': per_image_inference_time,
            'rpc_latency': rpc_latency,
            'data_transfer_size_bytes': data_transfer_size_bytes,
            'network_throughput_mbps': network_throughput_mbps,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'cpu_percent': cpu_percent,
            'accuracy': accuracy,
            'total_images': total_images,
            'predictions': str(predictions.tolist()) if predictions is not None else None,
            'actual_labels': str(actual_labels.tolist()) if actual_labels is not None else None
        }
        
        # Store in memory and write to CSV immediately
        self.metrics_data.append(metrics_record)
        self._write_to_csv(metrics_record)
        
        # Also log key metrics
        logging.info(f"[METRICS] {event_type} - Memory: {memory_percent}%, CPU: {cpu_percent}%, "
                    f"Inference time: {inference_time}, RPC latency: {rpc_latency}, "
                    f"Throughput: {network_throughput_mbps} Mbps")
    
    def _write_to_csv(self, record):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([record[key] for key in [
                'timestamp', 'datetime', 'rank', 'hostname', 'event_type',
                'batch_id', 'shard_id', 'image_id', 'tensor_shape', 
                'inference_time', 'per_image_inference_time', 'rpc_latency',
                'data_transfer_size_bytes', 'network_throughput_mbps',
                'memory_percent', 'memory_used_gb', 'cpu_percent',
                'accuracy', 'total_images', 'predictions', 'actual_labels'
            ]])
    
    def finalize(self):
        logging.info(f"Metrics saved to: {self.csv_file}")
        logging.info(f"Total metrics records: {len(self.metrics_data)}")


# split model into desired number of partitions
def split_model_into_n_shards(model: nn.Module, n: int) -> List[nn.Sequential]:
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

    elif isinstance(model, torchvision_models.Inception3):
        # InceptionV3: split all layers before final FC
        modules = list(model.children())[:-1]  # exclude fc
        split_idx = len(modules) // (n - 1)

        shards = []
        for i in range(n - 1):
            start = i * split_idx
            end = (i + 1) * split_idx if i < n - 2 else len(modules)
            shards.append(nn.Sequential(*modules[start:end]))

        final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            model.fc
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

    elif isinstance(model, torchvision_models.AlexNet):
        # AlexNet: features → classifier
        feature_layers = list(model.features.children())
        split_idx = len(feature_layers) // (n - 1)

        shards = []
        for i in range(n - 1):
            start = i * split_idx
            end = (i + 1) * split_idx if i < n - 2 else len(feature_layers)
            shards.append(nn.Sequential(*feature_layers[start:end]))

        final_block = nn.Sequential(
            nn.Flatten(),
            *list(model.classifier.children())
        )
        shards.append(final_block)
        return shards

    elif isinstance(model, torchvision_models.SqueezeNet):
        # SqueezeNet: features + classifier
        feature_layers = list(model.features.children())
        split_idx = len(feature_layers) // (n - 1)

        shards = []
        for i in range(n - 1):
            start = i * split_idx
            end = (i + 1) * split_idx if i < n - 2 else len(feature_layers)
            shards.append(nn.Sequential(*feature_layers[start:end]))

        final_block = nn.Sequential(
            model.classifier,  # includes Dropout, Conv2d, ReLU, AvgPool
            nn.Flatten()
        )
        shards.append(final_block)
        return shards

    elif isinstance(model, torchvision_models.VGG):
        # VGG16: features → avgpool → classifier
        feature_layers = list(model.features.children())
        split_idx = len(feature_layers) // (n - 2)

        shards = []
        for i in range(n - 2):
            start = i * split_idx
            end = (i + 1) * split_idx if i < n - 3 else len(feature_layers)
            shards.append(nn.Sequential(*feature_layers[start:end]))

        # Final 2 shards: avgpool + flatten, then classifier
        shards.append(nn.Sequential(
            model.avgpool,
            nn.Flatten()
        ))
        shards.append(model.classifier)
        return shards

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


class ShardWrapper(nn.Module):
    def __init__(self, submodule, shard_id, metrics_collector):
        super().__init__()
        self.module = submodule.to("cpu")
        self.shard_id = shard_id
        self.metrics_collector = metrics_collector

    def forward(self, x):  
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor but got {type(x)}")

        logging.info(f"[{socket.gethostname()}] Shard {self.shard_id} received tensor with shape: {x.shape}")

        x = x.to("cpu")
        
        # Start timing
        start_time = time.time()
        
        # Inference
        output = self.module(x).cpu()
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Collect metrics
        self.metrics_collector.collect_inference_metrics(
            event_type="shard_inference",
            shard_id=self.shard_id,
            tensor_shape=x.shape,
            inference_time=inference_time
        )

        logging.info(f"[{socket.gethostname()}] Shard {self.shard_id} output tensor shape: {output.shape}")
        return output

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


# Distributed model using pipeline parallelism
class DistributedModel(nn.Module):
    def __init__(self, model_type: str, num_splits: int, workers: List[str], num_classes: int = 10, metrics_collector=None):
        super(DistributedModel, self).__init__()
        self.num_splits = num_splits
        self.model_type = model_type
        self.workers = workers
        self.num_classes = num_classes
        self.worker_rrefs = []
        self.metrics_collector = metrics_collector

        # Load and prepare model
        if model_type == "mobilenetv2":
            model = torchvision_models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
            model.load_state_dict(torch.load("mobilenetv2_cifar10.pth", map_location="cpu"))

        elif model_type == "inceptionv3":
            model = torchvision_models.inception_v3(weights=None, aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load("inception_cifar10.pth", map_location="cpu"))

        elif model_type == "alexnet":
            model = torchvision_models.alexnet(weights=None) 
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  
            model.load_state_dict(torch.load("alexnet_cifar10_25epochs.pth", map_location=torch.device("cpu")))

        elif model_type == "resnet18":
            model = torchvision_models.resnet18(weights=None)  
            model.fc = nn.Linear(model.fc.in_features, num_classes) 
            model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location=torch.device("cpu")))

        elif model_type == "squeezenet":
            model = torchvision_models.squeezenet1_1(weights=None)  
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1)) 
            model.num_classes = num_classes
            model.load_state_dict(torch.load("squeezenet_cifar10.pth", map_location=torch.device("cpu")))

        elif model_type == "vgg16":
            model = torchvision_models.vgg16(weights=None)  
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  
            state_dict = torch.load("vgg16_cifar10_25epochs.pth", map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.eval()

        # Split and deploy to workers
        self.shards = split_model_into_n_shards(model, num_splits)

        for i, shard in enumerate(self.shards):
            worker_name = self.workers[i % len(self.workers)]  # round robin if num_splits > len(workers)
            # Pass shard_id and metrics_collector to workers
            rref = rpc.remote(worker_name, ShardWrapper, args=(shard, i, None))  # Workers will create their own metrics_collector
            self.worker_rrefs.append(rref)
    
    def forward(self, x, batch_id=None):
        x_rref = x
        
        # Start timing for end-to-end batch processing
        batch_start_time = time.time()
        
        for i, shard_rref in enumerate(self.worker_rrefs):
            logging.info(f"Sending tensor to shard {i} on worker {self.workers[i % len(self.workers)]}")
            x_rref = shard_rref.rpc_sync().forward(x_rref)
            
            # Stop passing to next shard if it's the last
            if i == len(self.worker_rrefs) - 1:
                break

        # Calculate total batch time
        batch_total_time = time.time() - batch_start_time
        
        # Collect batch-level metrics on master
        if self.metrics_collector:
            self.metrics_collector.collect_inference_metrics(
                event_type="batch_complete",
                batch_id=batch_id,
                tensor_shape=x.shape,
                inference_time=batch_total_time
            )

        return x_rref  # x_rref is now just a Tensor, not an RRef
        
    def parameter_rrefs(self):
        remote_params = []
        for rref in self.worker_rrefs:
            remote_params.extend(rref.remote().parameter_rrefs().to_here())
        return remote_params


def run_inference(rank, world_size, model_type, batch_size, num_micro_batches, num_classes, dataset, num_test_samples, model, num_splits, metrics_dir):
    """
    Main function to run distributed inference with metrics collection
    """

    connected = True
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector(rank, metrics_dir)
    
    # Record start of process
    metrics_collector.collect_inference_metrics("process_start")

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
        os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  # Typical WiFi interface name on Pis
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
            
            # Record RPC initialization
            metrics_collector.collect_inference_metrics("rpc_initialized")
            
            # Define worker names
            workers = [f"worker{i}" for i in range(1, world_size)]
            logger.info(f"Setting up model with workers: {workers}")
            
            # Create distributed model
            model = DistributedModel(
                model_type=model_type,
                num_splits=num_splits,
                workers=workers,
                num_classes=num_classes,
                metrics_collector=metrics_collector
            )
            logger.info("Distributed model created successfully")
            
            # Record model setup
            metrics_collector.collect_inference_metrics("model_setup_complete")
            
            # Load data
            logger.info(f"Loading {dataset} dataset")
            if dataset == 'cifar10':
                resize_dim = (299, 299) if model_type == "inceptionv3" else (224, 224)
                transform = transforms.Compose([
                    transforms.Resize(resize_dim),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                dataset_path = "/export/datasets/cifar10"
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
            
            # Record data loading
            metrics_collector.collect_inference_metrics("data_loaded")
            
            # Run inference
            logger.info("Starting inference...")
            start_time = time.time()
            
            total_images = 0
            num_correct = 0
            num_batches = (num_test_samples + batch_size - 1) // batch_size

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    # break if specified number of test images has been reached 
                    remaining = num_test_samples - total_images
                    if remaining <= 0:
                        break

                    # If current batch has more images than we need, trim it
                    if images.size(0) > remaining:
                        images = images[:remaining]
                        labels = labels[:remaining]

                    logger.info(f"Running inference on batch {i+1}/{num_batches} with shape: {images.shape}")
                    
                    # Run inference with batch tracking
                    output = model(images, batch_id=i)

                    logger.info(f"Received output shape: {output.shape}")

                    # log the predicted vs actual labels
                    _, predicted = torch.max(output.data, 1)
                    logger.info(f"Predicted: {predicted[:5]} | Actual: {labels[:5]}")

                    num_correct += (predicted == labels).sum().item()
                    total_images += len(images)
                    
                    # Calculate batch accuracy
                    batch_accuracy = (predicted == labels).sum().item() / len(labels) * 100.0
                    
                    # Record batch results
                    metrics_collector.collect_inference_metrics(
                        event_type="batch_result",
                        batch_id=i,
                        accuracy=batch_accuracy,
                        total_images=len(images),
                        predictions=predicted[:10],  # First 10 predictions
                        actual_labels=labels[:10]    # First 10 actual labels
                    )

            elapsed_time = time.time() - start_time
            final_accuracy = 1.0 * num_correct / total_images * 100.0
            
            logger.info(f"Inference completed on {total_images} images.")
            logger.info(f"Final Accuracy: {final_accuracy}") 
            logger.info(f"Inference completed in {elapsed_time:.4f} seconds")
            
            # Record final results
            metrics_collector.collect_inference_metrics(
                event_type="inference_complete",
                accuracy=final_accuracy,
                total_images=total_images,
                inference_time=elapsed_time
            )
            
            # Print some results
            if dataset == 'cifar10':
                _, predicted = torch.max(output.data, 1)
                logger.info(f"First few predictions: {predicted[:5]}")
                logger.info(f"First few actual labels: {labels[:5]}")
            
        except Exception as e:
            logger.error(f"Error in master node: {str(e)}", exc_info=True)
            metrics_collector.collect_inference_metrics("error", inference_time=None)
            
    else:  # Workers
        logger.info(f"Initializing worker node with rank {rank}")
        retry_count = 0
        max_retries = 30  # More retries
        connected = False
        
        while retry_count < max_retries and not connected:
            try:
                # Force workers to use WiFi interface
                os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
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
                
                # Record successful connection
                metrics_collector.collect_inference_metrics("worker_connected")
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Connection attempt {retry_count} failed: {str(e)}")
                metrics_collector.collect_inference_metrics("connection_failed")
                
                if retry_count >= max_retries:
                    logger.error(f"Worker {rank} failed to connect after {max_retries} attempts")
                    break  # Exit the loop instead of raising
                wait_time = 10 + (retry_count % 5)
                logger.info(f"Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                time.sleep(wait_time)

    if not connected and rank != 0:
        logger.error("Worker failed to connect to master node")
        metrics_collector.collect_inference_metrics("worker_failed")
        # Exit with a non-zero code so the shell script knows to retry
        sys.exit(1)
    
    # Record process completion
    metrics_collector.collect_inference_metrics("process_complete")
    
    # Only call shutdown if RPC was successfully initialized
    if rpc_initialized:
        logger.info("Waiting for RPC shutdown")
        try:
            rpc.shutdown()
            logger.info("RPC shutdown complete")
            metrics_collector.collect_inference_metrics("rpc_shutdown")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    else:
        logger.warning("RPC was never successfully initialized, skipping shutdown")
    
    # Finalize metrics collection
    metrics_collector.finalize()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Distributed DNN Inference on Raspberry Pi with Metrics")
    parser.add_argument("--rank", type=int, default=0, help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, help="World size (1 master + N workers)")
    parser.add_argument("--model", type=str, default="mobilenetv2", 
                        choices=["mobilenetv2", "inceptionv3", "alexnet", "resnet18", "squeezenet", "vgg16"],
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--micro-batches", type=int, default=2, help="Number of micro-batches for pipeline")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["cifar10", "dummy"],
                        help="Dataset to use for inference")
    parser.add_argument("--num-test-samples", type=int, default=10, help="Number of images to test on during inference")
    parser.add_argument("--num-partitions", type=int, default=2, help="Number of partitions to split the model into")
    parser.add_argument("--metrics-dir", type=str, default="./metrics", help="Directory to save metrics CSV files")
    
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
        num_test_samples=args.num_test_samples,
        model=args.model,
        num_splits=args.num_partitions,
        metrics_dir=args.metrics_dir,
    )


if __name__ == "__main__":
    main()