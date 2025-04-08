#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Model Evaluation Script

This script evaluates multiple deep learning models and collects performance metrics
including inference time, accuracy, CPU/GPU usage, and memory consumption.
"""

import os
import sys
import time
import argparse
import json
import csv
from datetime import datetime
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision.models as tv_models
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#############################################
# GLOBAL VARIABLES - Adjust as needed
#############################################

# Paths
DATA_ROOT = os.path.expanduser("~/datasets")
IMAGENET_PATH = os.path.join(DATA_ROOT, "imagenet")
VOC_PATH = os.path.join(DATA_ROOT, "voc2012")
COCO_PATH = os.path.join(DATA_ROOT, "coco")
MODEL_PATH = os.path.join(DATA_ROOT, "pretrained_models")

# Create directories if they don't exist
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Results directory
RESULTS_DIR = os.path.join(DATA_ROOT, "single_device_results")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

# Evaluation settings
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_NUM_INFERENCES = 100
DEFAULT_WARMUP_ITERATIONS = 10

# Model names
AVAILABLE_MODELS = [
    "mobilenetv2", 
    "deeplabv3", 
    "inception", 
    "resnet18", 
    "alexnet", 
    "vgg16", 
    "squeezenet"
]

#############################################
# Metrics Collection Utilities
#############################################

class MetricsCollector:
    """Class for collecting and aggregating performance metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.inference_times = []
        self.accuracies = []
        self.cpu_usages = []
        self.cpu_frequencies = []
        self.memory_usages = []
        self.gpu_usages = []
        self.gpu_frequencies = []
        self.gpu_memory_usages = []
    
    def start_inference(self):
        """Record metrics at the start of inference."""
        # CPU metrics
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        self.start_cpu_freq = cpu_freq.current if cpu_freq else 0
        self.start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        # GPU metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_gpu_time = torch.cuda.Event(enable_timing=True)
            self.end_gpu_time = torch.cuda.Event(enable_timing=True)
            self.start_gpu_time.record()
            
            if hasattr(torch.cuda, 'utilization'):
                self.start_gpu_util = torch.cuda.utilization()
            else:
                self.start_gpu_util = 0
                
            if hasattr(torch.cuda, 'mem_get_info'):
                free_mem, total_mem = torch.cuda.mem_get_info()
                self.start_gpu_mem = (total_mem - free_mem) / (1024 * 1024)  # MB
            else:
                self.start_gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    
    def end_inference(self, accuracy=None):
        """Record metrics at the end of inference."""
        # CPU metrics
        end_time = time.time()
        self.inference_times.append(end_time - self.start_time)
        
        end_cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usages.append((self.start_cpu_percent + end_cpu_percent) / 2)
        
        cpu_freq = psutil.cpu_freq()
        end_cpu_freq = cpu_freq.current if cpu_freq else 0
        self.cpu_frequencies.append((self.start_cpu_freq + end_cpu_freq) / 2)
        
        end_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        self.memory_usages.append((self.start_memory + end_memory) / 2)
        
        # GPU metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_gpu_time.record()
            torch.cuda.synchronize()
            self.inference_times[-1] = self.start_gpu_time.elapsed_time(self.end_gpu_time) / 1000  # convert to seconds
            
            if hasattr(torch.cuda, 'utilization'):
                end_gpu_util = torch.cuda.utilization()
                self.gpu_usages.append((self.start_gpu_util + end_gpu_util) / 2)
            else:
                self.gpu_usages.append(0)
                
            if hasattr(torch.cuda, 'mem_get_info'):
                free_mem, total_mem = torch.cuda.mem_get_info()
                end_gpu_mem = (total_mem - free_mem) / (1024 * 1024)  # MB
            else:
                end_gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
            self.gpu_memory_usages.append((self.start_gpu_mem + end_gpu_mem) / 2)
            
            # Estimate GPU frequency (this is a rough approximation)
            self.gpu_frequencies.append(0)  # Placeholder, hard to get programmatically
        
        # Record accuracy if provided
        if accuracy is not None:
            self.accuracies.append(accuracy)
    
    def get_summary(self):
        """Get summary statistics of all collected metrics."""
        summary = {}
        
        # Calculate averages for each metric
        if self.inference_times:
            summary["avg_inference_time_ms"] = np.mean(self.inference_times) * 1000  # convert to ms
            summary["std_inference_time_ms"] = np.std(self.inference_times) * 1000   # convert to ms
        
        if self.accuracies:
            summary["avg_accuracy"] = np.mean(self.accuracies)
            summary["std_accuracy"] = np.std(self.accuracies)
        
        if self.cpu_usages:
            summary["avg_cpu_usage_percent"] = np.mean(self.cpu_usages)
        
        if self.cpu_frequencies:
            summary["avg_cpu_freq_mhz"] = np.mean(self.cpu_frequencies)
        
        if self.memory_usages:
            summary["avg_memory_usage_mb"] = np.mean(self.memory_usages)
        
        if torch.cuda.is_available():
            if self.gpu_usages:
                summary["avg_gpu_usage_percent"] = np.mean(self.gpu_usages)
            
            if self.gpu_frequencies:
                summary["avg_gpu_freq_mhz"] = np.mean(self.gpu_frequencies)
            
            if self.gpu_memory_usages:
                summary["avg_gpu_memory_usage_mb"] = np.mean(self.gpu_memory_usages)
        
        return summary


#############################################
# Base Model Evaluator Class
#############################################

class ModelEvaluator:
    """Base class for model evaluation."""
    
    def __init__(self, model_name, batch_size=DEFAULT_BATCH_SIZE, 
                 num_workers=DEFAULT_NUM_WORKERS, num_inferences=DEFAULT_NUM_INFERENCES):
        """
        Initialize the model evaluator.
        
        Args:
            model_name (str): Name of the model
            batch_size (int): Batch size for evaluation
            num_workers (int): Number of workers for data loading
            num_inferences (int): Number of inferences to run
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_inferences = num_inferences
        self.device = DEVICE
        self.metrics = MetricsCollector()
        
        # Create model and move to device
        self.model = self._create_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Count parameters
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        
        # Prepare data
        self.data_loader = self._create_data_loader()
    
    def _create_model(self):
        """Create and return the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _create_model()")
    
    def _create_data_loader(self):
        """Create and return a data loader. To be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _create_data_loader()")
        
    def _download_dataset_if_needed(self, dataset_class, **kwargs):
        """Helper method to download a dataset if it's not already available."""
        try:
            # First try with download=False to check if it's already downloaded
            kwargs['download'] = False
            dataset_class(root=DATA_ROOT, **kwargs)
            logger.info(f"Dataset {dataset_class.__name__} already downloaded.")
            return True
        except (RuntimeError, FileNotFoundError) as e:
            logger.info(f"Dataset {dataset_class.__name__} not found: {e}. Downloading...")
            try:
                # Try again with download=True
                kwargs['download'] = True
                dataset_class(root=DATA_ROOT, **kwargs)
                logger.info(f"Successfully downloaded dataset {dataset_class.__name__}.")
                return True
            except Exception as e:
                logger.error(f"Error downloading dataset {dataset_class.__name__}: {e}")
                return False
    
    def _preprocess_input(self, inputs):
        """Preprocess the inputs if needed. Can be overridden by subclasses."""
        return inputs
    
    def _calculate_accuracy(self, outputs, targets):
        """Calculate accuracy. Can be overridden by subclasses."""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)
    
    def run_evaluation(self, warmup_iterations=DEFAULT_WARMUP_ITERATIONS):
        """Run the evaluation and collect metrics."""
        logger.info(f"Evaluating {self.model_name}...")
        
        # Warmup
        logger.info(f"Running {warmup_iterations} warmup iterations...")
        warmup_data = iter(self.data_loader)
        for _ in range(warmup_iterations):
            try:
                inputs, targets = next(warmup_data)
            except StopIteration:
                warmup_data = iter(self.data_loader)
                inputs, targets = next(warmup_data)
            
            inputs = self._preprocess_input(inputs)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            with torch.no_grad():
                _ = self.model(inputs)
        
        # Reset metrics collector
        self.metrics.reset()
        
        # Main evaluation loop
        logger.info(f"Running {self.num_inferences} inference iterations...")
        data_iter = iter(self.data_loader)
        for i in range(self.num_inferences):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                inputs, targets = next(data_iter)
            
            inputs = self._preprocess_input(inputs)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Start timing
            self.metrics.start_inference()
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(outputs, targets)
            
            # End timing and collect metrics
            self.metrics.end_inference(accuracy)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{self.num_inferences} iterations")
        
        # Get summary metrics
        self.results = self.metrics.get_summary()
        
        # Add model information
        self.results["model_name"] = self.model_name
        self.results["num_parameters"] = self.num_parameters
        self.results["device"] = str(self.device)
        self.results["gpu_name"] = GPU_NAME
        self.results["batch_size"] = self.batch_size
        self.results["num_inferences"] = self.num_inferences
        
        return self.results
    
    def save_results(self, output_dir=RESULTS_DIR, format="json"):
        """Save the evaluation results."""
        if not hasattr(self, 'results'):
            raise ValueError("No results to save. Run evaluation first.")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = os.path.join(output_dir, f"{self.model_name}_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=4)
        elif format.lower() == "csv":
            filename = os.path.join(output_dir, f"{self.model_name}_{timestamp}.csv")
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.results.keys())
                writer.writerow(self.results.values())
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {filename}")
        return filename


#############################################
# Model-Specific Classes
#############################################

class MobileNetV2Evaluator(ModelEvaluator):
    """Evaluator for MobileNetV2."""
    
    def _create_model(self):
        logger.info("Loading pretrained MobileNetV2 model...")
        # Set model weights download location
        weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1
        weights._download_to(MODEL_PATH)
        return tv_models.mobilenet_v2(weights=weights)
    
    def _create_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # MobileNetV2 is trained on ImageNet but we'll use CIFAR-10 for evaluation
        # since it's easier to download and still suitable for classification
        logger.info("Loading CIFAR-10 dataset for MobileNetV2 evaluation...")
        dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)


class DeepLabV3Evaluator(ModelEvaluator):
    """Evaluator for DeepLabV3."""
    
    def _create_model(self):
        logger.info("Loading pretrained DeepLabV3 model...")
        # DeepLabV3 with ResNet-50 backbone
        weights = tv_models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        weights._download_to(MODEL_PATH)
        return tv_models.segmentation.deeplabv3_resnet50(weights=weights)
    
    def _create_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # For segmentation, we'll use VOC if available (through torchvision.datasets.VOCSegmentation)
        # otherwise, we'll adapt CIFAR-10 for this test
        try:
            logger.info("Attempting to load VOCSegmentation dataset...")
            dataset = datasets.VOCSegmentation(root=DATA_ROOT, year='2012', 
                                             image_set='val', download=True,
                                             transform=transform)
        except Exception as e:
            logger.warning(f"Error loading VOCSegmentation: {e}")
            logger.warning("Using CIFAR-10 instead (not ideal for segmentation)")
            dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)
    
    def _calculate_accuracy(self, outputs, targets):
        """For segmentation models, use mIoU or another appropriate metric."""
        # This is a simplification; real segmentation evaluation would be more complex
        if isinstance(outputs, dict):
            outputs = outputs['out']
        
        # Handle both VOC segmentation and CIFAR-10 cases
        if isinstance(targets, tuple) and len(targets) == 2:
            # VOCSegmentation returns (image, target)
            targets = targets[1]
        
        # For VOC segmentation, compute pixel accuracy
        if outputs.shape[2:] != targets.shape[1:] and hasattr(targets, 'shape'):
            # Resize targets to match output size if needed
            targets = F.interpolate(targets.float().unsqueeze(1), size=outputs.shape[2:], 
                                   mode='nearest').long().squeeze(1)
        
        outputs = torch.argmax(outputs, dim=1)
        
        # Calculate accuracy
        if hasattr(targets, 'shape') and len(targets.shape) > 1 and targets.shape[0] == outputs.shape[0]:
            # For segmentation data
            correct = (outputs == targets).sum().item()
            total = targets.numel()
            return correct / total
        else:
            # Fallback for CIFAR-10
            _, predicted = torch.max(outputs.reshape(outputs.size(0), -1).mean(dim=1).unsqueeze(1), 1)
            correct = (predicted == targets).sum().item()
            return correct / targets.size(0)


class InceptionEvaluator(ModelEvaluator):
    """Evaluator for Inception v3."""
    
    def _create_model(self):
        logger.info("Loading pretrained Inception v3 model...")
        weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
        weights._download_to(MODEL_PATH)
        return tv_models.inception_v3(weights=weights)
    
    def _create_data_loader(self):
        # Inception v3 requires 299x299 input
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use CIFAR-10 for evaluation, with appropriate resizing
        logger.info("Loading CIFAR-10 dataset for Inception v3 evaluation...")
        dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)
    
    def _calculate_accuracy(self, outputs, targets):
        """Handle the auxiliary outputs from Inception."""
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Use only the primary output
        
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)


class ResNet18Evaluator(ModelEvaluator):
    """Evaluator for ResNet18."""
    
    def _create_model(self):
        logger.info("Loading pretrained ResNet18 model...")
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        weights._download_to(MODEL_PATH)
        return tv_models.resnet18(weights=weights)
    
    def _create_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Since we want to use built-in datasets, let's use CIFAR-100 for ResNet
        # CIFAR-100 has 100 classes which makes it a bit more suitable for evaluating ImageNet models
        logger.info("Loading CIFAR-100 dataset for ResNet18 evaluation...")
        dataset = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)


class AlexNetEvaluator(ModelEvaluator):
    """Evaluator for AlexNet."""
    
    def _create_model(self):
        logger.info("Loading pretrained AlexNet model...")
        weights = tv_models.AlexNet_Weights.IMAGENET1K_V1
        weights._download_to(MODEL_PATH)
        return tv_models.alexnet(weights=weights)
    
    def _create_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use CIFAR-10 for AlexNet as it's a standard classification dataset
        logger.info("Loading CIFAR-10 dataset for AlexNet evaluation...")
        dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)


class VGG16Evaluator(ModelEvaluator):
    """Evaluator for VGG16."""
    
    def _create_model(self):
        logger.info("Loading pretrained VGG16 model...")
        weights = tv_models.VGG16_Weights.IMAGENET1K_V1
        weights._download_to(MODEL_PATH)
        return tv_models.vgg16(weights=weights)
    
    def _create_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use STL10 dataset for VGG16, which has higher resolution images than CIFAR
        try:
            logger.info("Loading STL10 dataset for VGG16 evaluation...")
            dataset = datasets.STL10(root=DATA_ROOT, split='test', download=True, transform=transform)
        except Exception as e:
            logger.warning(f"Error loading STL10: {e}")
            logger.warning("Falling back to CIFAR-10")
            dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)


class SqueezeNetEvaluator(ModelEvaluator):
    """Evaluator for SqueezeNet."""
    
    def _create_model(self):
        logger.info("Loading pretrained SqueezeNet model...")
        weights = tv_models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        weights._download_to(MODEL_PATH)
        return tv_models.squeezenet1_1(weights=weights)
    
    def _create_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use Fashion-MNIST for SqueezeNet to provide variety in our dataset choices
        # SqueezeNet is designed to be resource-efficient, so it's interesting to test it on a simpler dataset
        try:
            logger.info("Loading FashionMNIST dataset for SqueezeNet evaluation...")
            # For FashionMNIST, we need to convert grayscale to RGB
            fashion_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to RGB
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            dataset = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=fashion_transform)
        except Exception as e:
            logger.warning(f"Error loading FashionMNIST: {e}")
            logger.warning("Falling back to CIFAR-10")
            dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)


#############################################
# Main execution
#############################################

def get_evaluator_class(model_name):
    """Get the evaluator class for a given model name."""
    evaluator_classes = {
        "mobilenetv2": MobileNetV2Evaluator,
        "deeplabv3": DeepLabV3Evaluator,
        "inception": InceptionEvaluator,
        "resnet18": ResNet18Evaluator,
        "alexnet": AlexNetEvaluator,
        "vgg16": VGG16Evaluator,
        "squeezenet": SqueezeNetEvaluator
    }
    
    if model_name.lower() not in evaluator_classes:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(evaluator_classes.keys())}")
    
    return evaluator_classes[model_name.lower()]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate deep learning models")
    
    parser.add_argument("--models", nargs='+', default=["resnet18"], 
                        choices=AVAILABLE_MODELS,
                        help=f"Models to evaluate. Available: {AVAILABLE_MODELS}")
    
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})")
    
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"Number of workers for data loading (default: {DEFAULT_NUM_WORKERS})")
    
    parser.add_argument("--num-inferences", type=int, default=DEFAULT_NUM_INFERENCES,
                        help=f"Number of inferences to run (default: {DEFAULT_NUM_INFERENCES})")
    
    parser.add_argument("--warmup-iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS,
                        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP_ITERATIONS})")
    
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR,
                        help=f"Directory to save results (default: {RESULTS_DIR})")
    
    parser.add_argument("--output-format", type=str, default="json", choices=["json", "csv"],
                        help="Format of output files (default: json)")
    
    parser.add_argument("--aggregate", action="store_true", 
                        help="Aggregate results from all models into a single file")
    
    return parser.parse_args()


def main():
    """Main function to run the evaluation."""
    args = parse_args()
    
    # Configure output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log system information
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {GPU_NAME}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Run evaluations for all specified models
    all_results = {}
    
    for model_name in args.models:
        try:
            # Get the evaluator class for this model
            EvaluatorClass = get_evaluator_class(model_name)
            
            # Create evaluator instance
            evaluator = EvaluatorClass(
                model_name=model_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_inferences=args.num_inferences
            )
            
            # Run evaluation
            results = evaluator.run_evaluation(warmup_iterations=args.warmup_iterations)
            
            # Save individual results
            evaluator.save_results(output_dir=args.output_dir, format=args.output_format)
            
            # Store for aggregation if needed
            all_results[model_name] = results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Aggregate results if requested
    if args.aggregate and all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.output_format.lower() == "json":
            filename = os.path.join(args.output_dir, f"aggregate_results_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(all_results, f, indent=4)
        elif args.output_format.lower() == "csv":
            filename = os.path.join(args.output_dir, f"aggregate_results_{timestamp}.csv")
            
            # Get all possible keys from all results
            all_keys = set()
            for result in all_results.values():
                all_keys.update(result.keys())
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header row with 'model_name' as the first column
                header = ['model_name'] + sorted(k for k in all_keys if k != 'model_name')
                writer.writerow(header)
                
                # Write a row for each model
                for model_name, result in all_results.items():
                    row = [model_name] + [result.get(k, '') for k in header[1:]]
                    writer.writerow(row)
        
        logger.info(f"Aggregate results saved to {filename}")
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()