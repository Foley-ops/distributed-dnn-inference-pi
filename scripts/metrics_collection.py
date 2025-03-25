#!/usr/bin/env python3

import time
import json
import os
import argparse
import numpy as np
import torch
import psutil
import platform
from typing import Dict, List, Tuple, Union, Optional

class MetricsCollector:
    """
    Class to collect performance metrics during distributed inference
    """
    def __init__(
        self, 
        output_dir: str = "results",
        experiment_name: Optional[str] = None,
        device_name: Optional[str] = None
    ):
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        self.device_name = device_name or platform.node()
        
        # Create results directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            "timestamp": time.time(),
            "device_name": self.device_name,
            "experiment_name": self.experiment_name,
            "system_info": self._get_system_info(),
            "inference_metrics": [],
            "memory_metrics": [],
            "energy_metrics": [],
            "network_metrics": []
        }
    
    def _get_system_info(self) -> Dict:
        """Collect system information"""
        try:
            # Get CPU information
            cpu_info = {
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
            }
            
            # Get memory information
            memory = psutil.virtual_memory()
            memory_info = {
                "total_memory_gb": round(memory.total / (1024**3), 2),
                "available_memory_gb": round(memory.available / (1024**3), 2)
            }
            
            # Get GPU information if available
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "cuda_available": True,
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                }
            else:
                gpu_info = {"cuda_available": False}
            
            return {**cpu_info, **memory_info, **gpu_info}
        
        except Exception as e:
            print(f"Error collecting system info: {e}")
            return {"error": str(e)}
    
    def record_inference_metrics(
        self, 
        model_name: str, 
        batch_size: int, 
        num_micro_batches: int,
        latency_ms: float, 
        throughput: float,
        rank: int = 0
    ) -> None:
        """Record inference performance metrics"""
        self.metrics["inference_metrics"].append({
            "timestamp": time.time(),
            "model_name": model_name,
            "batch_size": batch_size,
            "num_micro_batches": num_micro_batches,
            "rank": rank,
            "latency_ms": latency_ms,
            "throughput_samples_per_sec": throughput
        })
    
    def record_memory_usage(self, rank: int = 0) -> None:
        """Record memory usage during inference"""
        try:
            # CPU memory
            memory = psutil.virtual_memory()
            memory_used_gb = round((memory.total - memory.available) / (1024**3), 2)
            memory_percent = memory.percent
            
            # GPU memory if available
            gpu_memory_used_gb = None
            gpu_memory_percent = None
            
            if torch.cuda.is_available():
                # Get GPU memory usage
                gpu_mem_alloc = torch.cuda.memory_allocated(0)
                gpu_mem_reserved = torch.cuda.memory_reserved(0)
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
                
                gpu_memory_used_gb = round(gpu_mem_alloc / (1024**3), 2)
                gpu_memory_reserved_gb = round(gpu_mem_reserved / (1024**3), 2)
                gpu_memory_percent = round(100 * gpu_mem_alloc / total_gpu_memory, 2)
            
            self.metrics["memory_metrics"].append({
                "timestamp": time.time(),
                "rank": rank,
                "cpu_memory_used_gb": memory_used_gb,
                "cpu_memory_percent": memory_percent,
                "gpu_memory_used_gb": gpu_memory_used_gb,
                "gpu_memory_percent": gpu_memory_percent
            })
        
        except Exception as e:
            print(f"Error recording memory usage: {e}")
    
    def record_energy_usage(self, rank: int = 0) -> None:
        """Record energy usage (placeholder - would require additional hardware/software)"""
        # This would require additional hardware sensors or software
        # For Raspberry Pi, you could use external power monitors
        pass
    
    def record_network_metrics(self, rank: int = 0) -> None:
        """Record network usage during distributed inference"""
        try:
            # Get network I/O statistics
            net_io = psutil.net_io_counters()
            
            self.metrics["network_metrics"].append({
                "timestamp": time.time(),
                "rank": rank,
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            })
        
        except Exception as e:
            print(f"Error recording network metrics: {e}")
    
    def save_metrics(self) -> None:
        """Save collected metrics to JSON file"""
        output_file = os.path.join(
            self.output_dir, 
            f"{self.experiment_name}_{self.device_name}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to {output_file}")
    
    def print_summary(self) -> None:
        """Print a summary of the collected metrics"""
        print("\n" + "="*50)
        print(f"Experiment: {self.experiment_name}")
        print(f"Device: {self.device_name}")
        print("="*50)
        
        if self.metrics["inference_metrics"]:
            latencies = [m["latency_ms"] for m in self.metrics["inference_metrics"]]
            throughputs = [m["throughput_samples_per_sec"] for m in self.metrics["inference_metrics"]]
            
            print("\nInference Metrics:")
            print(f"  Number of measurements: {len(latencies)}")
            print(f"  Average latency: {np.mean(latencies):.2f} ms")
            print(f"  Minimum latency: {np.min(latencies):.2f} ms")
            print(f"  Maximum latency: {np.max(latencies):.2f} ms")
            print(f"  Average throughput: {np.mean(throughputs):.2f} samples/sec")
        
        if self.metrics["memory_metrics"]:
            print("\nMemory Metrics:")
            cpu_mem_percent = [m["cpu_memory_percent"] for m in self.metrics["memory_metrics"]]
            print(f"  Average CPU memory usage: {np.mean(cpu_mem_percent):.2f}%")
            
            if self.metrics["memory_metrics"][0]["gpu_memory_percent"] is not None:
                gpu_mem_percent = [m["gpu_memory_percent"] for m in self.metrics["memory_metrics"]]
                print(f"  Average GPU memory usage: {np.mean(gpu_mem_percent):.2f}%")
        
        print("="*50 + "\n")

def benchmark_inference(
    model_function,
    input_data,
    model_name: str,
    batch_size: int,
    num_micro_batches: int,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    metrics_collector: Optional[MetricsCollector] = None,
    rank: int = 0
) -> Dict:
    """
    Benchmark inference performance
    
    Args:
        model_function: Function that takes input_data and returns output
        input_data: Input data for the model
        model_name: Name of the model
        batch_size: Batch size
        num_micro_batches: Number of micro-batches for pipeline parallelism
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        metrics_collector: Optional metrics collector instance
        rank: Process rank in distributed setting
    
    Returns:
        Dictionary with benchmark results
    """
    # Warmup
    for _ in range(warmup_iterations):
        _ = model_function(input_data)
    
    # Benchmark
    latencies = []
    for i in range(num_iterations):
        # Start timing
        start_time = time.time()
        
        # Run inference
        _ = model_function(input_data)
        
        # Calculate latency
        end_time = time.time()
        latency_s = end_time - start_time
        latency_ms = latency_s * 1000
        latencies.append(latency_ms)
        
        # Calculate throughput
        throughput = batch_size / latency_s
        
        # Record metrics if collector provided
        if metrics_collector:
            metrics_collector.record_inference_metrics(
                model_name=model_name,
                batch_size=batch_size,
                num_micro_batches=num_micro_batches,
                latency_ms=latency_ms,
                throughput=throughput,
                rank=rank
            )
            
            # Also record resource usage
            metrics_collector.record_memory_usage(rank=rank)
            metrics_collector.record_network_metrics(rank=rank)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    avg_throughput = batch_size / (avg_latency / 1000)
    
    # Create results dictionary
    results = {
        "model_name": model_name,
        "batch_size": batch_size,
        "num_micro_batches": num_micro_batches,
        "latency_ms": {
            "avg": avg_latency,
            "min": min_latency,
            "max": max_latency,
            "p95": p95_latency,
            "p99": p99_latency
        },
        "throughput_samples_per_sec": avg_throughput
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metrics collection for distributed inference")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for metrics")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="Device name")
    args = parser.parse_args()
    
    # Create metrics collector
    collector = MetricsCollector(
        output_dir=args.output_dir,
        experiment_name=args.experiment,
        device_name=args.device
    )
    
    # Print system info
    print("System Information:")
    for key, value in collector.metrics["system_info"].items():
        print(f"  {key}: {value}")