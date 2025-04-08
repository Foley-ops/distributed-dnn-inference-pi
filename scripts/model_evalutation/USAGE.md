# Model Evaluation Script - Usage Guide

This guide explains how to use the model evaluation script to benchmark multiple deep learning models on a single device and collect performance metrics.

## Overview

The script supports evaluating the following models:
- MobileNetV2
- DeepLabV3
- Inception v3
- ResNet18
- AlexNet
- VGG16
- SqueezeNet

It measures and collects these metrics:
- Number of model parameters
- Average accuracy over inferences
- Average inference time per single inference
- Average CPU/GPU usage
- Average CPU/GPU frequency
- Average memory usage

## Prerequisites

```
pip install torch torchvision numpy psutil
```

The script will automatically download the pretrained models and datasets when first run.

## Basic Usage

To evaluate a single model with default settings:

```bash
python model_evaluation.py --models resnet18
```

To evaluate multiple models:

```bash
python model_evaluation.py --models resnet18 mobilenetv2 vgg16
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | Models to evaluate (space-separated) | `resnet18` |
| `--batch-size` | Batch size for evaluation | 16 |
| `--num-workers` | Number of workers for data loading | 4 |
| `--num-inferences` | Number of inferences to run | 100 |
| `--warmup-iterations` | Number of warmup iterations | 10 |
| `--output-dir` | Directory to save results | `./results` |
| `--output-format` | Format of output files (`json` or `csv`) | `json` |
| `--aggregate` | Aggregate results from all models into a single file | `False` |

## Examples

**Run evaluation with a larger batch size:**
```bash
python model_evaluation.py --models mobilenetv2 --batch-size 32
```

**Run fewer inferences with more warmup iterations:**
```bash
python model_evaluation.py --models alexnet --num-inferences 50 --warmup-iterations 20
```

**Evaluate all models and generate a CSV report:**
```bash
python model_evaluation.py --models mobilenetv2 deeplabv3 inception resnet18 alexnet vgg16 squeezenet --output-format csv --aggregate
```

**Run a quick test with minimal iterations:**
```bash
python model_evaluation.py --models squeezenet --num-inferences 10 --warmup-iterations 2
```