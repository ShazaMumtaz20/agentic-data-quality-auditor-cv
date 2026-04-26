"""
Model Evaluation Module
Trains lightweight baseline models on cleaned vs uncleaned datasets to justify preprocessing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import cv2
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import time


class ImageDataset(Dataset):
    """Simple dataset class for image classification."""
    
    def __init__(self, dataset_path: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            dataset_path: Path to dataset directory (ImageFolder structure)
            transform: Optional torchvision transforms
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Load images and labels
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all images from the dataset."""
        class_dirs = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            
            # Get all image files
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in valid_extensions:
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 if needed
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        # Convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class SimpleCNN(nn.Module):
    """
    Lightweight CNN baseline model for evaluation.
    Small enough to train quickly but capable of learning basic patterns.
    """
    
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtModel, self).__init__()
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.model.classifier[2] = nn.Linear(
            self.model.classifier[2].in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


class RegNetModel(nn.Module):
    def __init__(self, num_classes):
        super(RegNetModel, self).__init__()
        self.model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class NFNetModel(nn.Module):
    def __init__(self, num_classes):
        super(NFNetModel, self).__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "NFNet requires the optional 'timm' package. Install it with: pip install timm"
            ) from exc

        self.model = timm.create_model('nfnet_f0', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class ModelEvaluator:
    """
    Evaluates model performance on cleaned vs uncleaned datasets.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluator.

        Args:
            device: 'cuda' or 'cpu'
        """
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Install CUDA-enabled PyTorch or use device='cpu'.")

        self.device = torch.device(device)
        print(f"Using device: {self.device}")

    def _create_model(self, model_type: str, num_classes: int) -> nn.Module:
        """Create the requested evaluation model."""
        if model_type == "cnn":
            return SimpleCNN(num_classes=num_classes)
        if model_type == "resnet":
            return ResNetModel(num_classes)
        if model_type == "efficientnet":
            return EfficientNetModel(num_classes)
        if model_type == "convnext":
            return ConvNeXtModel(num_classes)
        if model_type == "regnet":
            return RegNetModel(num_classes)
        if model_type == "nfnet":
            return NFNetModel(num_classes)
        raise ValueError(f"Invalid model type: {model_type}")

    
    def train_model(self, dataset_path: str, num_epochs: int = 5, 
                   batch_size: int = 32, learning_rate: float = 0.001, model_type: str = "cnn") -> Dict:
        """
        Train a lightweight model on the dataset.
        
        Args:
            dataset_path: Path to dataset directory
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            model_type: Type of architecture to use
            
        Returns:
            Dictionary with training results and metrics
        """
        print(f"\nTraining model on: {dataset_path}")
        
        # Create dataset
        dataset = ImageDataset(dataset_path)
        
        if len(dataset) == 0:
            return {
                'success': False,
                'error': 'No images found in dataset'
            }
        
        num_classes = len(dataset.class_to_idx)
        print(f"  Dataset: {len(dataset)} images, {num_classes} classes")
        print(f"  Classes: {list(dataset.class_to_idx.keys())}")
        
        # Split dataset (80% train, 20% test)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = self._create_model(model_type=model_type, num_classes=num_classes)

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"  Training for {num_epochs} epochs...")
        train_losses = []
        train_accuracies = []
        
        best_loss = float('inf')
        patience = 10
        counter = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered")
                break

            print(f"    Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        # Evaluation
        print("  Evaluating on test set...")
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100 * test_correct / test_total
        
        # Calculate per-class metrics
        class_report = classification_report(
            all_labels, all_preds,
            target_names=[dataset.idx_to_class[i] for i in range(num_classes)],
            output_dict=True,
            zero_division=0
        )
        conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Confusion Matrix: {conf_matrix}")
        
        return {
            'success': True,
            'model_type': model_type,
            'test_accuracy': test_accuracy,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'confusion_matrix': conf_matrix,
            'train_accuracies': train_accuracies,
            'train_losses': train_losses,
            'num_classes': num_classes,
            'num_images': len(dataset),
            'class_report': class_report,
            'class_names': list(dataset.class_to_idx.keys())
        }
    
    def compare_datasets(self, original_path: str, cleaned_path: str,
                    num_epochs: int = 5, batch_size: int = 32,
                    model_type: str = "cnn") -> Dict:
        """
        Compare model performance on original vs cleaned datasets.
        
        Args:
            original_path: Path to original (uncleaned) dataset
            cleaned_path: Path to cleaned dataset
            num_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION: CLEANED vs UNCLEANED DATASETS")
        print("="*80)
        
        # Train on original dataset
        print("\n[1/2] Training on ORIGINAL (uncleaned) dataset...")
        original_results = self.train_model(original_path, num_epochs, batch_size, model_type=model_type)
        
        if not original_results['success']:
            return {
                'success': False,
                'error': f"Failed to train on original dataset: {original_results.get('error')}"
            }
        
        # Train on cleaned dataset
        print("\n[2/2] Training on CLEANED dataset...")
        cleaned_results = self.train_model(cleaned_path, num_epochs, batch_size, model_type=model_type)
        
        if not cleaned_results['success']:
            return {
                'success': False,
                'error': f"Failed to train on cleaned dataset: {cleaned_results.get('error')}"
            }
        
        # Compare results
        accuracy_improvement = cleaned_results['test_accuracy'] - original_results['test_accuracy']
        improvement_percent = (accuracy_improvement / original_results['test_accuracy']) * 100 if original_results['test_accuracy'] > 0 else 0
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(f"Original Dataset:")
        print(f"  Images: {original_results['num_images']}")
        print(f"  Test Accuracy: {original_results['test_accuracy']:.2f}%")
        print(f"  Macro F1: {original_results['macro_f1']:.4f}")
        print(f"  Confusion Matrix: {original_results['confusion_matrix']}")
        print(f"\nCleaned Dataset:")
        print(f"  Images: {cleaned_results['num_images']}")
        print(f"  Test Accuracy: {cleaned_results['test_accuracy']:.2f}%")
        print(f"  Macro F1: {cleaned_results['macro_f1']:.4f}")
        print(f"  Confusion Matrix: {cleaned_results['confusion_matrix']}")
        print(f"\nImprovement:")
        print(f"  Accuracy: {original_results['test_accuracy']:.2f}% → {cleaned_results['test_accuracy']:.2f}%")
        print(f"  Improvement: {accuracy_improvement:+.2f}% ({improvement_percent:+.1f}% relative)")
        
        if accuracy_improvement > 0:
            print(f"\n✓ CLEANED DATASET PERFORMS BETTER - Preprocessing justified!")
        elif accuracy_improvement < 0:
            print(f"\n⚠ CLEANED DATASET PERFORMS WORSE - May need to review preprocessing")
        else:
            print(f"\n→ NO SIGNIFICANT DIFFERENCE")
        
        return {
            'success': True,
            'original': original_results,
            'cleaned': cleaned_results,
            'improvement': accuracy_improvement,
            'improvement_percent': improvement_percent,
            'justified': accuracy_improvement > 0
        }
