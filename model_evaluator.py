import sys
if sys.platform == "win32" and sys.exec_prefix.endswith(":"):
    sys.exec_prefix += "\\"

import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


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

    def __init__(self, device: str = "cpu"):
        """
        Initialize evaluator.

        Args:
            device: 'cuda' or 'cpu'
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = "cpu"

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

    def save_model_checkpoint(self, model: nn.Module, checkpoint_path: str):
        """Persist a model checkpoint to disk so one frozen model can be reused across stages."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'device': str(self.device),
        }, checkpoint_path)
        return str(checkpoint_path)

    def load_model_checkpoint(self, model: nn.Module, checkpoint_path: str):
        """Load a saved model checkpoint into the current model instance."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model

    def evaluate_stage_predictions(self, model: nn.Module, image_paths: list[str], true_label_ids: list[int],
                                  stage_name: str, output_dir: str = 'results', batch_size: int = 32):
        """Run a frozen model across a stage's test set and save paired predictions/labels."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        predictions = []
        correct_flags = []

        model.eval()
        with torch.no_grad():
            for image_path, true_label in zip(image_paths, true_label_ids):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = image.astype(np.float32) / 255.0
                tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
                logits = model(tensor)
                pred = int(torch.argmax(logits, dim=1).item())
                predictions.append(pred)
                correct_flags.append(int(pred == true_label))

        payload = {
            'stage': stage_name,
            'image_count': len(image_paths),
            'image_paths': [str(p) for p in image_paths],
            'predictions': predictions,
            'true_labels': [int(v) for v in true_label_ids],
            'correct': correct_flags,
            'accuracy': float(np.mean(correct_flags)) if correct_flags else 0.0,
        }

        with open(out_dir / f'{stage_name}_predictions.json', 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        return payload

    @staticmethod
    def mcnemar_exact_p_value(correct_a: list[int], correct_b: list[int]):
        """Exact two-sided McNemar p-value from paired correctness flags using the binomial sign test."""
        if len(correct_a) != len(correct_b):
            raise ValueError('Paired correctness arrays must have the same length')
        b = 0
        c = 0
        for a, b_flag in zip(correct_a, correct_b):
            if a == 1 and b_flag == 0:
                b += 1
            elif a == 0 and b_flag == 1:
                c += 1
        n = b + c
        if n == 0:
            return 1.0
        p_both = 0.5 ** n
        tail = 0.0
        for k in range(0, min(b, c) + 1):
            tail += math.comb(n, k) * p_both
        p_value = 2.0 * tail
        return min(1.0, float(p_value))

    
    def train_model(self, dataset_path: str, num_epochs: int = 5,
                   batch_size: int = 32, learning_rate: float = 0.001, model_type: str = "cnn",
                   early_stopping_patience: int = 5, early_stopping_min_delta: float = 0.0,
                   validation_ratio: float = 0.2, early_stopping_metric: str = "accuracy") -> Dict:
        """
        Train a lightweight model on the dataset.
        
        Args:
            dataset_path: Path to dataset directory
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            model_type: Type of architecture to use
            early_stopping_patience: Number of epochs without validation loss improvement before stopping
            early_stopping_min_delta: Minimum validation loss improvement required to reset patience
            validation_ratio: Fraction of the training split reserved for validation
            early_stopping_metric: Metric to monitor for stopping ('loss', 'accuracy', or 'both')
            
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

        early_stopping_metric = early_stopping_metric.lower().strip()
        if early_stopping_metric not in {'loss', 'accuracy', 'both'}:
            raise ValueError("early_stopping_metric must be one of: 'loss', 'accuracy', 'both'")
        
        num_classes = len(dataset.class_to_idx)
        print(f"  Dataset: {len(dataset)} images, {num_classes} classes")
        print(f"  Classes: {list(dataset.class_to_idx.keys())}")
        
        # Split dataset into train/validation/test so early stopping monitors a held-out signal.
        train_pool_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_pool_size
        train_pool, test_dataset = torch.utils.data.random_split(
            dataset, [train_pool_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        if len(train_pool) > 1 and validation_ratio > 0:
            val_size = max(1, int(round(len(train_pool) * validation_ratio)))
            if val_size >= len(train_pool):
                val_size = len(train_pool) - 1
            train_size = len(train_pool) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_pool, [train_size, val_size],
                generator=torch.Generator().manual_seed(43)
            )
        else:
            train_dataset = train_pool
            val_dataset = None
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
        
        # Create model
        model = self._create_model(model_type=model_type, num_classes=num_classes)

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"  Training for {num_epochs} epochs...")
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        early_stopped = False
        best_epoch = 0
        
        best_val_loss = float('inf')
        best_val_acc = -float('inf')
        best_state_dict = None
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

            if val_loader is not None:
                model.eval()
                val_running_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        val_running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_loss = val_running_loss / len(val_loader)
                val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                loss_improved = val_loss < (best_val_loss - early_stopping_min_delta)
                acc_improved = val_acc > (best_val_acc + early_stopping_min_delta)

                if early_stopping_metric == 'loss':
                    improved = loss_improved
                elif early_stopping_metric == 'accuracy':
                    improved = acc_improved
                else:
                    improved = loss_improved or acc_improved

                if loss_improved:
                    best_val_loss = val_loss
                if acc_improved:
                    best_val_acc = val_acc

                if improved:
                    best_epoch = epoch + 1
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    counter = 0
                else:
                    counter += 1

                print(
                    f"    Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%"
                    f" | ValLoss={val_loss:.4f}, ValAcc={val_acc:.2f}%"
                )

                if counter >= early_stopping_patience:
                    early_stopped = True
                    print(
                        f"    Early stopping triggered at epoch {epoch+1}; best epoch {best_epoch} "
                        f"(monitor={early_stopping_metric})"
                    )
                    break
            else:
                print(f"    Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        elif val_loader is None:
            best_epoch = len(train_losses)
        
        # Evaluation
        print("  Evaluating on test set...")
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        test_image_paths = [dataset.images[i] for i in test_dataset.indices]
        test_label_ids = [int(dataset.labels[i]) for i in test_dataset.indices]

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
        correct_flags = [int(pred == label) for pred, label in zip(all_preds, all_labels)]
        
        # Compute metrics on the full class space, even when a tiny subset leaves some
        # classes absent in the test split. This avoids target_name length mismatches.
        labels = list(range(num_classes))
        class_report = classification_report(
            all_labels, all_preds,
            labels=labels,
            target_names=[dataset.idx_to_class[i] for i in labels],
            output_dict=True,
            zero_division=0
        )
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=labels).tolist()
        macro_f1 = f1_score(all_labels, all_preds, average='macro', labels=labels, zero_division=0)
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
            'val_accuracies': val_accuracies,
            'val_losses': val_losses,
            'early_stopping': {
                'enabled': val_loader is not None,
                'patience': early_stopping_patience,
                'min_delta': early_stopping_min_delta,
                'monitor': early_stopping_metric,
                'best_epoch': best_epoch,
                'stopped_early': early_stopped,
            },
            'num_classes': num_classes,
            'num_images': len(dataset),
            'class_report': class_report,
            'class_names': list(dataset.class_to_idx.keys()),
            'predictions': [int(p) for p in all_preds],
            'true_labels': [int(y) for y in all_labels],
            'correct': correct_flags,
            'test_image_paths': test_image_paths,
            'test_label_ids': test_label_ids,
        }

    @staticmethod
    def _prediction_file_name(stage_name: str) -> str:
        normalized = stage_name.strip().lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        aliases = {
            'no_preprocessing': 'no_preprocessing',
            'stage_1': 'stage_1',
            'stage_2': 'stage_2',
            'stage_3': 'stage_3',
            'stage_4': 'stage_4',
            'original': 'no_preprocessing',
            'cleaned': 'stage_1',
        }
        return aliases.get(normalized, normalized) + '_predictions.json'

    def save_prediction_artifact(self, result_payload: Dict, stage_name: str, output_dir: str = 'results') -> Dict:
        """Persist a model prediction artifact with paired correctness flags for downstream McNemar tests."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        payload = {
            'stage': stage_name,
            'image_count': len(result_payload.get('test_image_paths', [])),
            'image_paths': [str(p) for p in result_payload.get('test_image_paths', [])],
            'true_labels': [int(v) for v in result_payload.get('true_labels', [])],
            'predictions': [int(v) for v in result_payload.get('predictions', [])],
            'correct': [int(v) for v in result_payload.get('correct', [])],
            'accuracy': float(np.mean(result_payload.get('correct', []))) if result_payload.get('correct') else 0.0,
        }

        file_name = self._prediction_file_name(stage_name)
        with open(output_path / file_name, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        return payload
    
    def compare_datasets(self, original_path: str, cleaned_path: str,
                    num_epochs: int = 5, batch_size: int = 32,
                    model_type: str = "cnn", early_stopping_patience: int = 5,
                    early_stopping_min_delta: float = 0.0,
                    validation_ratio: float = 0.2, early_stopping_metric: str = "accuracy") -> Dict:
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
        original_results = self.train_model(
            original_path,
            num_epochs,
            batch_size,
            model_type=model_type,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            validation_ratio=validation_ratio,
            early_stopping_metric=early_stopping_metric,
        )
        
        if not original_results['success']:
            return {
                'success': False,
                'error': f"Failed to train on original dataset: {original_results.get('error')}"
            }
        
        # Train on cleaned dataset
        print("\n[2/2] Training on CLEANED dataset...")
        cleaned_results = self.train_model(
            cleaned_path,
            num_epochs,
            batch_size,
            model_type=model_type,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            validation_ratio=validation_ratio,
            early_stopping_metric=early_stopping_metric,
        )
        
        if not cleaned_results['success']:
            return {
                'success': False,
                'error': f"Failed to train on cleaned dataset: {cleaned_results.get('error')}"
            }
        
        result_dir = Path('results')
        result_dir.mkdir(parents=True, exist_ok=True)
        self.save_prediction_artifact(original_results, 'No preprocessing', output_dir=str(result_dir))
        self.save_prediction_artifact(cleaned_results, 'Stage 1', output_dir=str(result_dir))

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
