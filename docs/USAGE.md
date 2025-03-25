# Distributed DNN Inference - Usage Guide

This guide explains how to run distributed inference using the pipeline parallelism framework.

## Basic Usage

The distributed inference system uses PyTorch's RPC framework to distribute model layers across multiple devices. Follow these steps to run inference:

### 1. Start Workers First

On each worker Raspberry Pi, run the inference script with its rank and world size:

```bash
# On worker1 (rank 1)
python rpc_layer_split.py --rank 1 --world-size 3

# On worker2 (rank 2)
python rpc_layer_split.py --rank 2 --world-size 3
```

Make sure to set the environment variables first:

```bash
export MASTER_ADDR=master
export MASTER_PORT=29500
```

### 2. Start Master Last

On the master Raspberry Pi, run:

```bash
# On master (rank 0)
python rpc_layer_split.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --micro-batches 4
```

## Command Line Options

The main script accepts the following command line arguments:

- `--rank`: Rank of the current process (0 for master, 1+ for workers)
- `--world-size`: Total number of processes (1 master + N workers)
- `--model`: Model architecture to use (choices: mobilenetv2, squeezenet, efficientnet_b0)
- `--batch-size`: Batch size for inference
- `--micro-batches`: Number of micro-batches for pipeline parallelism
- `--num-classes`: Number of output classes (default: 1000)
- `--dataset`: Dataset to use for inference (choices: cifar10, mnist)

## Example Configurations

### Simple Setup (3 Devices)

```bash
# On worker1
python rpc_layer_split.py --rank 1 --world-size 3

# On worker2
python rpc_layer_split.py --rank 2 --world-size 3

# On master
python rpc_layer_split.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 16 --micro-batches 4
```

### Larger Setup (4+ Devices)

For more devices, adjust the world size and ranks accordingly:

```bash
# On worker1
python rpc_layer_split.py --rank 1 --world-size 5

# On worker2
python rpc_layer_split.py --rank 2 --world-size 5

# On worker3
python rpc_layer_split.py --rank 3 --world-size 5

# On worker4
python rpc_layer_split.py --rank 4 --world-size 5

# On master
python rpc_layer_split.py --rank 0 --world-size 5 --model efficientnet_b0 --batch-size 8 --micro-batches 8
```

## Performance Metrics Collection

To collect performance metrics during inference, use the metrics collection script:

```bash
# On master
python scripts/metrics_collection.py --experiment mobilenet_test --output-dir results
```

The metrics will be saved to JSON files in the specified output directory.

## Visualizing Results

You can visualize the collected metrics using the provided utility:

```bash
python scripts/plot_metrics.py --input results/mobilenet_test_*.json --output plots
```

This will generate performance comparison plots in the specified output directory.

## Tuning Parameters

To achieve the best performance, consider tuning these parameters:

1. **Batch Size**: Larger batch sizes increase throughput but also increase memory usage.

2. **Number of Micro-batches**: More micro-batches can improve pipeline utilization but add overhead.

3. **Model Size**: Choose an appropriate model size for your hardware:
   - MobileNetV2: Lightweight, good for Raspberry Pi
   - SqueezeNet: Very small, fastest inference
   - EfficientNet-B0: More accurate but heavier

## Troubleshooting

### Common Issues

1. **Connection Timeouts**:
   - Ensure all workers are started before the master
   - Check network connectivity between devices
   - Increase the RPC timeout in the code if needed

2. **Out of Memory Errors**:
   - Reduce batch size
   - Use a smaller model
   - Ensure no other memory-intensive processes are running

3. **RPC Errors**:
   - Ensure all devices have the same PyTorch version
   - Verify hostname resolution works correctly
   - Check that all processes have different ranks

### Performance Issues

If inference is slower than expected:

1. Monitor CPU/memory usage with:
   ```bash
   htop
   ```

2. Check network performance with:
   ```bash
   iperf3 -c worker1
   ```

3. Monitor temperature (Raspberry Pis may throttle when hot):
   ```bash
   vcgencmd measure_temp
   ```