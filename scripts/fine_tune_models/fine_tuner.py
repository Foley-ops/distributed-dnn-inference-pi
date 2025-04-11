#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Fine-tuning Script for Edge Device Benchmarking

This script fine-tunes pretrained models on their target datasets to achieve
higher accuracy for benchmarking on Raspberry Pi devices.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torchvision.models as tv_models
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_ROOT = os.path.expanduser("~/datasets")
MODEL_PATH = os.path.join(DATA_ROOT, "fine_tuned_models")
os.makedirs(MODEL_PATH, exist_ok=True)

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available models
AVAILABLE_MODELS = [
    "mobilenetv2", 
    "inception", 
    "resnet18", 
    "alexnet", 
    "vgg16", 
    "squeezenet"
]

# Training settings
DEFAULT_BATCH_SIZE = 64  # Larger for faster training
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LR = 0.001

class ModelFinetuner:
    """Base class for fine-tuning models."""
    
    def __init__(self, model_name, batch_size=DEFAULT_BATCH_SIZE, 
                 num_epochs=DEFAULT_NUM_EPOCHS, learning_rate=DEFAULT_LR, num_workers=4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.device = DEVICE
        
        # Create data loaders first to get num_classes
        self.train_loader, self.val_loader = self._create_data_loaders()
        self.num_classes = self._get_num_classes()
        
        # Create and adapt model
        self.model = self._create_model()
        self.model = self._adapt_model(self.model, self.num_classes)
        self.model.to(self.device)
        
        # Set up optimizer and criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.get_trainable_parameters(), lr=self.learning_rate)
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _create_model(self):
        """Create and return the base model. To be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _create_model()")
    
    def _adapt_model(self, model, num_classes):
        """Adapt the model's classifier to match the target dataset's number of classes."""
        raise NotImplementedError("Subclass must implement _adapt_model()")
    
    def _create_data_loaders(self):
        """Create and return training and validation data loaders. To be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _create_data_loaders()")
    
    def _get_num_classes(self):
        """Get the number of classes from the dataset."""
        dataset = self.train_loader.dataset
        
        if hasattr(dataset, 'classes'):
            return len(dataset.classes)
        
        # If dataset is a subset from random_split
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
            return len(dataset.dataset.classes)
        
        # Try to infer from the first batch
        try:
            data_iter = iter(self.train_loader)
            _, targets = next(data_iter)
            return len(torch.unique(targets))
        except:
            # Default to 10 classes (typical for CIFAR-10, MNIST, etc.)
            logger.warning("Could not determine number of classes, defaulting to 10")
            return 10
    
    def get_trainable_parameters(self):
        """Return parameters that should be trained. Can be overridden to implement layer freezing."""
        return self.model.parameters()
    
    def train_one_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for inputs, targets in pbar:
            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_name == "inception":
                # Inception returns auxiliary outputs when training
                outputs, aux_outputs = self.model(inputs)
                loss1 = self.criterion(outputs, targets)
                loss2 = self.criterion(aux_outputs, targets)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def validate(self):
        """Validate the model and compute metrics."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                if self.model_name == "inception":
                    # During validation, only use primary output
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                else:
                    outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Train the model for the specified number of epochs with early stopping."""
        logger.info(f"Starting fine-tuning of {self.model_name} for {self.num_epochs} epochs")
        logger.info(f"Training on device: {self.device}")
        
        best_acc = 0.0
        best_model_state = None
        best_epoch = 0
        
        # Early stopping parameters
        patience = 3  # Stop if no improvement for 3 epochs
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Train one epoch
            train_loss = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}")
            
            # Check if model improved
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # Increment patience counter
                
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered. No improvement for {patience} epochs.")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch+1} with validation accuracy: {best_acc:.4f}")
        
        return best_acc
    
    def save_model(self):
        """Save the fine-tuned model."""
        model_filename = os.path.join(MODEL_PATH, f"{self.model_name}_finetuned.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'accuracy': self.val_accuracies[-1] if self.val_accuracies else 0,
        }, model_filename)
        
        logger.info(f"Model saved to {model_filename}")
        return model_filename


class MobileNetV2Finetuner(ModelFinetuner):
    """Fine-tuner for MobileNetV2."""
    
    def _create_model(self):
        logger.info("Loading pretrained MobileNetV2 model...")
        return tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    def _adapt_model(self, model, num_classes):
        logger.info(f"Adapting MobileNetV2 to {num_classes} classes")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    
    def _create_data_loaders(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use CIFAR-10 for MobileNetV2 - good balance of simplicity and challenge
        logger.info("Loading CIFAR-10 dataset for MobileNetV2...")
        train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        
        return train_loader, val_loader
    
    def get_trainable_parameters(self):
        # Freeze first 10 layers to preserve low-level features
        params_to_update = []
        freeze_layers = 10
        
        for i, param in enumerate(self.model.features.parameters()):
            if i >= freeze_layers:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
        
        # Always train the classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            params_to_update.append(param)
            
        return params_to_update


class InceptionFinetuner(ModelFinetuner):
    """Fine-tuner for Inception v3."""
    
    def _create_model(self):
        logger.info("Loading pretrained Inception v3 model...")
        model = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = True  # Enable auxiliary outputs during training
        return model
    
    def _adapt_model(self, model, num_classes):
        logger.info(f"Adapting Inception v3 to {num_classes} classes")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        aux_in_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)
        
        return model
    
    def _create_data_loaders(self):
        # Inception requires 299x299 input
        transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Try STL10 first (higher resolution), fall back to CIFAR-10
        try:
            logger.info("Loading STL10 dataset for Inception...")
            train_dataset = datasets.STL10(root=DATA_ROOT, split='train', download=True, transform=transform_train)
            val_dataset = datasets.STL10(root=DATA_ROOT, split='test', download=True, transform=transform_val)
        except Exception as e:
            logger.warning(f"Error loading STL10: {e}")
            logger.warning("Falling back to CIFAR-10")
            train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_val)
        
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        
        return train_loader, val_loader
    
    def get_trainable_parameters(self):
        # Freeze the first several layers to preserve ImageNet features
        params_to_update = []
        
        # Freeze first 6 blocks
        for name, param in self.model.named_parameters():
            if not name.startswith(('Mixed_6', 'Mixed_7', 'fc', 'AuxLogits')):
                param.requires_grad = False
            else:
                param.requires_grad = True
                params_to_update.append(param)
        
        return params_to_update


class ResNet18Finetuner(ModelFinetuner):
    """Fine-tuner for ResNet18."""
    
    def _create_model(self):
        logger.info("Loading pretrained ResNet18 model...")
        return tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    
    def _adapt_model(self, model, num_classes):
        logger.info(f"Adapting ResNet18 to {num_classes} classes")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    
    def _create_data_loaders(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use CIFAR-10
        logger.info("Loading CIFAR-10 dataset for ResNet18...")
        train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        
        return train_loader, val_loader
    
    def get_trainable_parameters(self):
        # Freeze early layers to preserve general features
        params_to_update = []
        
        # Freeze first layer and first block
        for name, param in self.model.named_parameters():
            if not name.startswith(('layer2', 'layer3', 'layer4', 'fc')):
                param.requires_grad = False
            else:
                param.requires_grad = True
                params_to_update.append(param)
        
        return params_to_update


class AlexNetFinetuner(ModelFinetuner):
    """Fine-tuner for AlexNet."""
    
    def _create_model(self):
        logger.info("Loading pretrained AlexNet model...")
        return tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1)
    
    def _adapt_model(self, model, num_classes):
        logger.info(f"Adapting AlexNet to {num_classes} classes")
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model
    
    def _create_data_loaders(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use CIFAR-10 directly
        logger.info("Using CIFAR-10 dataset for AlexNet...")
        train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        
        return train_loader, val_loader
    
    def get_trainable_parameters(self):
        # AlexNet is older and simpler - fine-tune the entire model except first few conv layers
        params_to_update = []
        
        # Only freeze the first convolutional layer
        for i, param in enumerate(self.model.features.parameters()):
            if i < 2:  # First conv layer has weights and bias (2 params)
                param.requires_grad = False
            else:
                param.requires_grad = True
                params_to_update.append(param)
        
        # Always train the classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            params_to_update.append(param)
            
        return params_to_update


class VGG16Finetuner(ModelFinetuner):
    """Fine-tuner for VGG16."""
    
    def _create_model(self):
        logger.info("Loading pretrained VGG16 model...")
        return tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    
    def _adapt_model(self, model, num_classes):
        logger.info(f"Adapting VGG16 to {num_classes} classes")
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model
    
    def _create_data_loaders(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use CIFAR-10 directly - more reliable
        logger.info("Using CIFAR-10 dataset for VGG16...")
        train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        
        return train_loader, val_loader
    
    def get_trainable_parameters(self):
        # VGG16 is deep - freeze most of the convolutional layers for efficiency
        params_to_update = []
        
        # Freeze first 10 convolutional layers (out of 13)
        layer_count = 0
        for name, param in self.model.features.named_parameters():
            if 'weight' in name:
                layer_count += 1
                
            if layer_count <= 10:
                param.requires_grad = False
            else:
                param.requires_grad = True
                params_to_update.append(param)
        
        # Always train the classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            params_to_update.append(param)
            
        return params_to_update


class SqueezeNetFinetuner(ModelFinetuner):
    """Fine-tuner for SqueezeNet."""
    
    def _create_model(self):
        logger.info("Loading pretrained SqueezeNet model...")
        return tv_models.squeezenet1_1(weights=tv_models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    
    def _adapt_model(self, model, num_classes):
        logger.info(f"Adapting SqueezeNet to {num_classes} classes")
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        return model
    
    def _create_data_loaders(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Use CIFAR-10 directly for reliability
        logger.info("Using CIFAR-10 dataset for SqueezeNet...")
        train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        
        return train_loader, val_loader
        
    def get_trainable_parameters(self):
        # SqueezeNet is already small, fine-tune most of it
        params_to_update = []
        
        # Freeze only the first two fire modules
        for name, param in self.model.named_parameters():
            if not name.startswith(('features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'classifier')):
                param.requires_grad = False
            else:
                param.requires_grad = True
                params_to_update.append(param)
        
        return params_to_update


def get_finetuner_class(model_name):
    """Get the finetuner class for a given model name."""
    finetuner_classes = {
        "mobilenetv2": MobileNetV2Finetuner,
        "inception": InceptionFinetuner,
        "resnet18": ResNet18Finetuner,
        "alexnet": AlexNetFinetuner,
        "vgg16": VGG16Finetuner,
        "squeezenet": SqueezeNetFinetuner
    }
    
    if model_name.lower() not in finetuner_classes:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(finetuner_classes.keys())}")
    
    return finetuner_classes[model_name.lower()]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune models for edge device benchmarking")
    
    parser.add_argument("--models", nargs='+', default=["resnet18"], 
                        choices=AVAILABLE_MODELS,
                        help=f"Models to fine-tune. Available: {AVAILABLE_MODELS}")
    
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS,
                        help=f"Number of epochs to train (default: {DEFAULT_NUM_EPOCHS})")
    
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    
    parser.add_argument("--num-workers", type=int, default=4,
                    help="Number of workers for data loading (default: 4)")

    return parser.parse_args()


def main():
    """Main function to run the fine-tuning."""
    args = parse_args()
    
    # Log system information
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Fine-tune each specified model
    for model_name in args.models:
        try:
            logger.info(f"Starting fine-tuning for {model_name}")
            
            # Get the finetuner class for this model
            FinetunerClass = get_finetuner_class(model_name)
            
            # Create finetuner instance
            finetuner = FinetunerClass(
                model_name=model_name,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                num_workers=args.num_workers
            )
            
            # Train the model
            best_acc = finetuner.train()
            
            # Save the fine-tuned model
            model_path = finetuner.save_model()
            
            logger.info(f"Fine-tuning complete for {model_name}. Best accuracy: {best_acc:.4f}")
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error fine-tuning {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("Fine-tuning complete for all models")


if __name__ == "__main__":
    main()