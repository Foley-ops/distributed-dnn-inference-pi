# Raspberry Pi Setup for Distributed DNN Inference

This guide explains how to set up multiple Raspberry Pi devices for distributed deep neural network inference using PyTorch's RPC framework.

## Hardware Requirements

- Multiple Raspberry Pi devices (4B with at least 4GB RAM recommended)
- Ethernet cables for networking (for lowest latency)
- Power supplies for each Raspberry Pi
- Optional: Cooling solutions (heatsinks, fans)

## Network Setup

### 1. Connect devices to the same network

All Raspberry Pi devices should be connected to the same local network, preferably via Ethernet for lowest latency.

### 2. Configure static IP addresses

For reliable connections, assign static IP addresses to each Raspberry Pi:

1. Edit the dhcpcd configuration file:
   ```
   sudo nano /etc/dhcpcd.conf
   ```

2. Add the following at the end of the file (adjust as needed):
   ```
   interface eth0
   static ip_address=192.168.1.X/24  # Use different X for each Pi
   static routers=192.168.1.1
   static domain_name_servers=192.168.1.1 8.8.8.8
   ```

3. Restart the dhcpcd service:
   ```
   sudo service dhcpcd restart
   ```

### 3. Configure hostname resolution

1. Edit the hosts file on each Raspberry Pi:
   ```
   sudo nano /etc/hosts
   ```

2. Add entries for all Raspberry Pi devices:
   ```
   127.0.0.1       localhost
   192.168.1.100   master
   192.168.1.101   worker1
   192.168.1.102   worker2
   # Add more as needed
   ```

## Software Setup

### 1. Install required packages

On each Raspberry Pi, install the necessary packages:

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install dependencies
sudo apt install -y build-essential cmake git python3-dev python3-pip

# Install Python packages
pip3 install --upgrade pip
pip3 install numpy psutil
```

### 2. Install PyTorch

For Raspberry Pi, we'll use the PyTorch wheel optimized for ARM:

```bash
# Install PyTorch for Raspberry Pi
pip3 install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0
```

Note: Check the PyTorch website for the latest available versions compatible with Raspberry Pi.

### 3. Clone the repository

```bash
git clone https://github.com/yourusername/distributed-dnn-inference-pi.git
cd distributed-dnn-inference-pi
```

### 4. Install project requirements

```bash
pip3 install -r requirements.txt
```

## SSH Key Setup (Optional but Recommended)

To allow easier automation between devices, set up SSH keys:

1. Generate SSH key on the master node:
   ```
   ssh-keygen -t rsa -b 4096
   ```

2. Copy the key to each worker:
   ```
   ssh-copy-id pi@worker1
   ssh-copy-id pi@worker2
   # Repeat for each worker
   ```

## Testing the Setup

Verify that the nodes can communicate with each other:

```bash
# From master node
ping worker1
ping worker2

# Test SSH connection
ssh pi@worker1 hostname
ssh pi@worker2 hostname
```

## Environment Variables Setup

Create a script to set environment variables before running distributed training:

```bash
# setup_env.sh
export MASTER_ADDR=master
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=eth0  # Use the network interface you're using
export TP_SOCKET_IFNAME=eth0
```

Source this file before running:

```bash
source setup_env.sh
```

## Troubleshooting

### Network Connectivity Issues

- Ensure all devices are on the same subnet
- Check firewall settings (allow ports 29500-29510)
- Verify hostname resolution works

### PyTorch RPC Issues

- Make sure MASTER_ADDR and MASTER_PORT are set correctly
- Check that all workers have the same version of PyTorch
- Increase timeout values for slower connections

### Performance Issues

- Monitor CPU temperature with `vcgencmd measure_temp`
- Consider adding cooling solutions if temperatures exceed 80°C
- Reduce model complexity or batch size if memory issues occur