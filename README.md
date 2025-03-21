# Distributed DNN Inference on Raspberry Pi Cluster

**Project:** Optimizing Distributed Deep Neural Network (DNN) Inference for Low Latency and Energy Efficiency on Resource-Constrained Edge Devices.

---

## Project Description

This research project aims to develop and evaluate novel methods for optimizing **distributed deep neural network (DNN) inference** to achieve low latency and energy efficiency specifically on resource-constrained **Raspberry Pi 4B (4GB RAM)** devices.

We start by manually splitting neural network models (layer-wise model parallelism) across multiple Raspberry Pis. Eventually, our goal is to dynamically partition these models based on real-time metrics such as device CPU load, memory usage, and network latency.

---

## Goals & Objectives

- Develop manual and dynamic model partitioning methods.
- Measure performance metrics per device:
  - Throughput
  - Network latency & delay
  - End-to-end inference time
  - Inference time per image/batch
  - CPU usage, frequency, memory usage
  - Accuracy
- Optimize for low latency and minimal energy consumption.
- Implement adaptive logic for dynamic layer assignment.

---

## Models and Datasets

Currently using pretrained models and datasets:

### Models:
- MobileNetV2
- MobileNetV3 (Small)
- SqueezeNet (v1.1)
- ShuffleNetV2 (x0.5)
- EfficientNet-B0

### Datasets:
- MNIST / FashionMNIST
- CIFAR-10 / CIFAR-100

---

## Project Structure

```
distributed-dnn-inference-pi/
├── README.md
├── requirements.txt
├── rpc_layer_split.py           # Main script for distributed inference using PyTorch RPC
├── models/                      # Pretrained models and related scripts
│   ├── mobilenetv2/
│   ├── squeezenet/
│   └── efficientnet_b0/
├── data/                        # Datasets used for inference
│   ├── cifar10/
│   └── mnist/
├── scripts/                     # Utility scripts for performance metrics and setup
│   └── metrics_collection.py
└── docs/                        # Documentation and instructions
    ├── SETUP.md                 # How to set up Raspberry Pi devices and environment
    └── USAGE.md                 # How to run inference and collect metrics
```

---

## Setup & Installation

Detailed instructions in: [docs/SETUP.md](docs/SETUP.md)

Quick overview:

1. Install Ubuntu 22.04 on all Raspberry Pi 4Bs.
2. Install PyTorch and dependencies.
3. Configure network connectivity between Pis.
4. Clone this repository and install Python dependencies:

```
git clone <repo-url>
cd distributed-dnn-inference-pi
pip install -r requirements.txt
```

---

## Usage

Detailed instructions in: [docs/USAGE.md](docs/USAGE.md)

Quick overview:

- To start distributed inference (manual partition):

```
python rpc_layer_split.py
```

- To collect performance metrics:

```
python scripts/metrics_collection.py
```

---

## Performance Metrics Collected

| Metric                       | Description                          |
|------------------------------|--------------------------------------|
| Model                        | Name of the DNN model used           |
| Device                       | Hostname or identifier of each Pi    |
| Throughput                   | Images processed per second          |
| Network Latency              | Latency between devices              |
| End-to-end time              | Total inference pipeline time        |
| Model Parameters             | Total parameters in the model        |
| Average Accuracy             | Accuracy over test set               |
| Inference Time (per image)   | Avg. inference time per image        |
| Inference Time (per batch)   | Avg. inference time per batch        |
| CPU Usage & Frequency        | Avg. CPU load and frequency (MHz)    |
| Memory Usage                 | Avg. RAM used during inference (%)   |
| Batch Size                   | Number of images processed per batch |
| Layers per Device            | Layers assigned to each Pi           |

---

## Developer Workflow

We use GitHub Projects to manage tasks, code reviews, and feature tracking. Contributions should follow this workflow:

1. Create a feature branch:

```
git checkout -b feature/your-feature-name
```

2. Commit your changes clearly:

```
git commit -m "Add feature X for Y"
```

3. Push your branch and open a Pull Request (PR) on GitHub:

```
git push origin feature/your-feature-name
```

4. Once reviewed, merge into `main`.

---

## Future Work

- Implement dynamic model partitioning based on runtime metrics.
- Evaluate and integrate additional models and datasets.
- Explore quantization and other optimization methods to enhance performance.

---

## Contributing

- Keep your code clean and readable.
- Document your changes thoroughly.
- Be sure your updates pass all tests before merging.

---

## Resources & References

- [PyTorch RPC Documentation](https://pytorch.org/docs/stable/rpc.html)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [Official Raspberry Pi Documentation](https://www.raspberrypi.org/documentation/)

---

## Contact

- **Project Lead:** [Your Name]([your-contact-info])
- **Lab:** [CloudSys Lab] at [University of Texas at San Antonio]

For questions or contributions, open an issue or contact the maintainers.

---




