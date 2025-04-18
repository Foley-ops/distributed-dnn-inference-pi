# ✅ Pi1 Client for 3-node distributed inference
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import zmq
import pickle
import time
import psutil
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

# CONFIG
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PI2_IP = "10.100.230.4"  # IP address of Pi2
PI2_PORT = 5555
NUM_BATCHES = 5  # Number of batches to process

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class Pi1Model(nn.Module):
    def __init__(self):
        super(Pi1Model, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Taking first 1/3 of the model
        self.features = nn.Sequential(*list(mobilenet.features.children())[:7])
        
    def forward(self, x):
        return self.features(x)

model = Pi1Model().to(DEVICE).eval()

# Create ZMQ socket for sending data to Pi2
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.setsockopt(zmq.LINGER, 1000)  # Set linger period for socket closure
socket.connect(f"tcp://{PI2_IP}:{PI2_PORT}")

# Establish a socket for acknowledgments from Pi3
feedback_context = zmq.Context()
feedback_socket = feedback_context.socket(zmq.PULL)
feedback_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10-second timeout for receiving acks
feedback_socket.bind(f"tcp://*:5556")  # Use a different port for feedback

# Metrics containers
metrics = {
    "batch_sizes": [],
    "feature_sizes": [],
    "data_sizes": [],
    "inference_times": [],
    "serialization_times": [],
    "network_send_times": [],
    "network_ack_times": [],
    "buffer_times": [],
    "end_to_end_times": [],
    "round_trip_delays": [],
    "cpu_usages": [],
    "cpu_freqs": [],
    "mem_usages": [],
    "timestamps": []
}

print(f"[Pi1] Starting distributed inference on {DEVICE}")
print(f"[Pi1] Connecting to Pi2 at {PI2_IP}:{PI2_PORT}")

start_total = time.time()

for idx, (images, labels) in enumerate(loader):
    if idx >= NUM_BATCHES:
        break
    
 
    
    # System metrics
    metrics["cpu_usages"].append(psutil.cpu_percent(interval=None))
    metrics["cpu_freqs"].append(psutil.cpu_freq().current if psutil.cpu_freq() else 0)
    metrics["mem_usages"].append(psutil.virtual_memory().percent)
    
    # Buffer time (time spent waiting before processing)
    buffer_start = time.time()
    # Simulate any preprocessing or buffering here if needed
    buffer_end = time.time()
    metrics["buffer_times"].append(buffer_end - buffer_start)
    
    # Inference time
    images = images.to(DEVICE)
    inference_start = time.time()
    with torch.no_grad():
        features = model(images)
    inference_end = time.time()
    metrics["inference_times"].append(inference_end - inference_start)
    metrics["feature_sizes"].append(features.element_size() * features.nelement())
    
    # Serialization time
    serialize_start = time.time()
    batch_start = time.time()
    data = pickle.dumps((features.cpu(), labels, batch_start))  # Include origin timestamp
    serialize_end = time.time()
    metrics["serialization_times"].append(serialize_end - serialize_start)
    metrics["data_sizes"].append(len(data))
    
    # Network send time
    send_start = time.time()
    socket.send(data)
    send_end = time.time()
    metrics["network_send_times"].append(send_end - send_start)
    
    # Wait for acknowledgment from Pi3 to measure round-trip time
    try:
        ack_start = time.time()
        ack = feedback_socket.recv()
        ack_end = time.time()
        metrics["network_ack_times"].append(ack_end - ack_start)
        ack_data = pickle.loads(ack)
        
        # Calculate round-trip delay across all 3 Pis
        round_trip_delay = ack_end - batch_start
        metrics["round_trip_delays"].append(round_trip_delay)
        
        print(f"[Pi1] Batch {idx+1} processed by Pi2 and Pi3")
        print(f"[Pi1] Round-trip time: {round_trip_delay:.4f}s")
        print(f"[Pi1] Accuracy: {ack_data['accuracy']:.2f}%")
    except zmq.error.Again:
        print(f"[Pi1] No acknowledgment received for batch {idx+1}")
        metrics["network_ack_times"].append(None)
        metrics["round_trip_delays"].append(None)
    
    batch_end = time.time()
    metrics["end_to_end_times"].append(batch_end - batch_start)
    
    print(f"[Pi1] Sent batch {idx+1}/{NUM_BATCHES} - Size: {len(data)/1024:.2f} KB")

end_total = time.time()

# Calculate aggregate metrics
def safe_mean(lst):
    filtered = [x for x in lst if x is not None]
    return sum(filtered) / len(filtered) if filtered else 0

metrics_summary = {
    "device": str(DEVICE),
    "total_batches": len(metrics["batch_sizes"]),
    "total_images": sum(metrics["batch_sizes"]),
    "total_time": end_total - start_total,
    "avg_batch_size": np.mean(metrics["batch_sizes"]),
    "avg_feature_size_kb": np.mean([s/1024 for s in metrics["feature_sizes"]]),
    "avg_data_size_kb": np.mean([s/1024 for s in metrics["data_sizes"]]),
    "avg_buffer_time": np.mean(metrics["buffer_times"]),
    "avg_inference_time": np.mean(metrics["inference_times"]),
    "avg_inference_time_per_image": np.mean(metrics["inference_times"]) / BATCH_SIZE,
    "avg_serialization_time": np.mean(metrics["serialization_times"]),
    "avg_network_send_time": np.mean(metrics["network_send_times"]),
    "avg_network_ack_time": safe_mean(metrics["network_ack_times"]),
    "avg_round_trip_delay": safe_mean(metrics["round_trip_delays"]),
    "avg_end_to_end_time": np.mean(metrics["end_to_end_times"]),
    "avg_cpu_usage": np.mean(metrics["cpu_usages"]),
    "avg_cpu_freq_mhz": np.mean(metrics["cpu_freqs"]),
    "avg_mem_usage": np.mean(metrics["mem_usages"])
}

# Save detailed metrics to file
with open("pi1_detailed_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("pi1_summary_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

# Generate visualizations
if metrics["batch_sizes"]:
    batches = range(1, len(metrics["batch_sizes"]) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Inference Time
    plt.subplot(2, 2, 1)
    plt.plot(batches, metrics["inference_times"], marker='o', color='blue', label='Inference Time')
    plt.title("Pi1 Inference Time per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Time (s)")
    plt.grid(True)
    
    # Subplot 2: Round-trip Delay
    plt.subplot(2, 2, 2)
    plt.plot(batches, metrics["round_trip_delays"], marker='s', color='red', label='Round-trip Delay')
    plt.title("Round-trip Delay (Pi1→Pi2→Pi3→Pi1) per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Time (s)")
    plt.grid(True)
    
    # Subplot 3: Processing Breakdown
    plt.subplot(2, 2, 3)
    plt.bar(batches, metrics["buffer_times"], color='green', label='Buffer')
    plt.bar(batches, metrics["inference_times"], bottom=metrics["buffer_times"], 
            color='orange', label='Inference')
    plt.bar(batches, metrics["serialization_times"], 
            bottom=[a+b for a,b in zip(metrics["buffer_times"], metrics["inference_times"])], 
            color='blue', label='Serialization')
    plt.title("Pi1 Processing Time Breakdown")
    plt.xlabel("Batch")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    
    # Subplot 4: Data Size
    plt.subplot(2, 2, 4)
    plt.plot(batches, [s/1024 for s in metrics["data_sizes"]], marker='*', color='purple', label='Data Size')
    plt.title("Data Size Sent to Pi2")
    plt.xlabel("Batch")
    plt.ylabel("Size (KB)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("pi1_performance_metrics.png")
    print("[Pi1] Performance visualization saved to pi1_performance_metrics.png")

# ✅ Comprehensive Metrics Summary
print("\n📊 [Pi1] Performance Metrics:")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Device: {DEVICE}")
print(f"Total Time: {end_total - start_total:.4f}s")
print(f"Images Processed: {sum(metrics['batch_sizes'])}")

print("\n⏱️ Timing Breakdown per Batch:")
print(f"Buffer Time: {metrics_summary['avg_buffer_time']:.4f}s")
print(f"Inference Time: {metrics_summary['avg_inference_time']:.4f}s")
print(f"Serialization Time: {metrics_summary['avg_serialization_time']:.4f}s")
print(f"Network Send Time: {metrics_summary['avg_network_send_time']:.4f}s")
print(f"End-to-End Time: {metrics_summary['avg_end_to_end_time']:.4f}s")

print("\n🔄 Network Metrics:")
print(f"Avg Data Size: {metrics_summary['avg_data_size_kb']:.2f} KB")
print(f"Network Send Time: {metrics_summary['avg_network_send_time']:.4f}s")
print(f"Round-Trip Time (Pi1→Pi2→Pi3→Pi1): {metrics_summary['avg_round_trip_delay']:.4f}s")

print("\n💻 System Metrics:")
print(f"Avg CPU Usage: {metrics_summary['avg_cpu_usage']:.2f}%")
print(f"Avg CPU Frequency: {metrics_summary['avg_cpu_freq_mhz']:.2f} MHz")
print(f"Avg Memory Usage: {metrics_summary['avg_mem_usage']:.2f}%")

print("\n📋 Detailed metrics saved to pi1_detailed_metrics.json")
print("📋 Summary metrics saved to pi1_summary_metrics.json")