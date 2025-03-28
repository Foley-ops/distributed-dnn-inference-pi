#!/usr/bin/env python3
import os
import time
import torch.distributed.rpc as rpc
import logging
import socket
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_test_function():
    return "Hello from worker"

def run_simple_test(rank, world_size):
    hostname = socket.gethostname()
    logger.info(f"Starting on {hostname} with rank {rank}")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get master address and port from .env file
    master_addr = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '29555')
    logger.info(f"Using master address: {master_addr} and port: {master_port}")
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Configure network interfaces
    if rank == 0:
        os.environ['GLOO_SOCKET_IFNAME'] = 'enp6s0'
        os.environ['TP_SOCKET_IFNAME'] = 'enp6s0'
        logger.info("Master using interface enp6s0")
    else:
        os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'
        os.environ['TP_SOCKET_IFNAME'] = 'wlan0'
        logger.info("Worker using interface wlan0")
    
    # Initialize RPC
    try:
        rpc.init_rpc(
            name=f"worker{rank}" if rank > 0 else "master",
            rank=rank,
            world_size=world_size
        )
        logger.info(f"RPC initialized successfully on rank {rank}")
        
        # Test RPC communication
        if rank == 0:
            # Wait for workers to initialize
            time.sleep(10)
            for i in range(1, world_size):
                try:
                    logger.info(f"Master trying to call worker{i}")
                    ret = rpc.rpc_sync(f"worker{i}", simple_test_function, args=())
                    logger.info(f"Got response from worker{i}: {ret}")
                except Exception as e:
                    logger.error(f"Failed to call worker{i}: {e}")
        else:
            # Workers just wait
            logger.info(f"Worker {rank} waiting")
            time.sleep(60)
        
        # Shutdown RPC
        logger.info(f"Rank {rank} shutting down RPC")
        rpc.shutdown()
        logger.info(f"Rank {rank} shutdown complete")
    except Exception as e:
        logger.error(f"Error on rank {rank}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=3)
    args = parser.parse_args()
    
    run_simple_test(args.rank, args.world_size)