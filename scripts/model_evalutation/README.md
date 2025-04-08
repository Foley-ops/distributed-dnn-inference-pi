# Deep Learning Model Performance Benchmark

A comprehensive benchmark suite for evaluating the performance of multiple deep learning models on a single device.

## Overview

This benchmark toolkit allows you to measure and compare key performance metrics of popular deep learning models running on single devices, focusing on:

- Number of model parameters
- Average accuracy over inferences
- Average inference time per inference
- CPU/GPU usage
- CPU/GPU frequencies
- Memory usage

This data can be used to compare against distributed inference performance to determine whether distribution is beneficial for your specific use case.

## Files in this Repository

- `model_evaluation.py` - The main Python script for running model evaluations
- `custom_model_example.py` - Example of how to extend the framework with a custom model
- `batch_run.sh` - Bash script for running all models sequentially and generating reports
- `README.md` - This documentation file

## Supported Models

The framework currently supports these popular pretrained models from PyTorch:

- MobileNetV2 (evaluated on CIFAR-10)
- DeepLabV3 (evaluated on VOCSegmentation or CIFAR-10 as fallback)
- Inception v3 (evaluated on CIFAR-10)
- ResNet18 (evaluated on CIFAR-100)
- AlexNet (evaluated on CIFAR-10)
- VGG16 (evaluated on STL10 or CIFAR-10 as fallback)
- SqueezeNet (evaluated on FashionMNIST or CIFAR-10 as fallback)

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- psutil

Install dependencies:

```bash
pip install torch torchvision numpy psutil
```

**Note:** All pretrained models and datasets will be automatically downloaded when first running the evaluation script. Models are downloaded to the PyTorch cache directory, and datasets are downloaded to the specified `DATA_ROOT` directory.

### Basic Usage

Run a single model evaluation:

```bash
python model_evaluation.py --models resnet18
```

Run multiple models and generate a comparative report:

```bash
python model_evaluation.py --models mobilenetv2 vgg16 resnet18 --aggregate
```

Run the batch script to evaluate all models:

```bash
bash batch_run.sh
```

## Extending the Framework

You can easily add new models by creating a new evaluator class that inherits from the `ModelEvaluator` base class. See `custom_model_example.py` for a complete example.

## Output

Results are saved in JSON or CSV format in the specified output directory. When run with the `--aggregate` flag, a combined report is generated with metrics from all evaluated models.

## NFS Support

The framework is designed to work with NFS storage, with configurable global variables at the top of the script for dataset paths and other shared resources.

## Notes

- For optimal results, minimize background processes during evaluation
- Use warmup iterations to ensure the system is in a steady state before measurement
- Run multiple evaluations to account for run-to-run variance
- Adjust batch size according to your device's capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

If you encounter GPU-related errors, try:
- Reducing batch size
- Ensuring you have the correct CUDA version installed
- Checking for sufficient GPU memory

For dataset-related issues, verify the paths in the global variables section.