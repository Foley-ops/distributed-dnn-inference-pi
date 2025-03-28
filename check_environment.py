#!/usr/bin/env python3

import pkg_resources
import sys
import os

def get_installed_packages():
    # Get all installed packages
    installed_packages = pkg_resources.working_set
    
    # Convert to a sorted list
    installed_packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])
    
    # Print environment information
    print(f"Python version: {sys.version}")
    print(f"Executable path: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Print virtual environment info if available
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"Virtual environment: {sys.prefix}")
    else:
        print("Not running in a virtual environment")
    
    # Print package list
    print("\nInstalled packages:")
    for package in installed_packages_list:
        print(f"  {package}")
    
    # Check specifically for PyTorch and related packages
    print("\nPyTorch-related packages:")
    pytorch_packages = [pkg for pkg in installed_packages_list 
                      if any(name in pkg.lower() for name in ["torch", "torchvision", "torchaudio"])]
    
    if pytorch_packages:
        for package in pytorch_packages:
            print(f"  {package}")
    else:
        print("  No PyTorch-related packages found")
        
    # Try to import torch and get more details
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            
        # Check if distributed is available
        if hasattr(torch, 'distributed'):
            print("torch.distributed is available")
            if hasattr(torch.distributed, 'is_available') and callable(torch.distributed.is_available):
                print(f"torch.distributed.is_available(): {torch.distributed.is_available()}")
    except ImportError:
        print("\nPyTorch is not installed")
    except Exception as e:
        print(f"\nError getting PyTorch details: {str(e)}")

if __name__ == "__main__":
    get_installed_packages()