import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from tqdm import tqdm
import pandas as pd
from src.model import EngagementClassifier, CombinedLoss, analyze_feature_importance

def create_balanced_sampler(dataset):
    # Convert float labels to long (integer) type
    labels = [int(label.item()) for _, label in dataset]
    class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    weights = 1. / class_counts.float()
    sample_weights = torch.tensor([weights[label] for label in labels])
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

class TrainingTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.thresholds = []
        
    def update(self, train_loss, val_loss, val_acc, lr, threshold):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.thresholds.append(threshold)
    
    def plot_training_progress(self):
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 4, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.subplot(1, 4, 3)
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.title('Learning Rate over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        plt.subplot(1, 4, 4)
        plt.plot(self.thresholds, label='Decision Threshold')
        plt.title('Decision Threshold Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

def plot_feature_importances(importances):
    plt.figure(figsize=(12, 8))
    features = list(importances.keys())
    values = list(importances.values())
    
    # Sort features by importance
    sorted_idx = np.argsort(values)
    features = [features[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    # Group features by type for visualization
    feature_colors = {
        'engagement': 'skyblue',
        'comment': 'lightgreen',
        'profile': 'salmon'
    }
    
    colors = []
    for feature in features:
        if feature in ['comment_burst_frequency', 'avg_comment_length', 'comment_timing_variance']:
            colors.append(feature_colors['comment'])
        elif feature in ['follower_following_ratio', 'avg_post_frequency', 'profile_completion_score']:
            colors.append(feature_colors['profile'])
        else:
            colors.append(feature_colors['engagement'])
    
    plt.barh(features, values, color=colors)
    plt.title('Feature Importance Analysis')
    plt.xlabel('Importance Score')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=label) 
                      for label, color in feature_colors.items()]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

def train_model(train_loader, val_loader, epochs=50):
    model = EngagementClassifier(input_features=15)  # Updated for new feature count
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0005,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    tracker = TrainingTracker()
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (features, labels) in enumerate(progress_bar):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
            })
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                val_loss += criterion(outputs, labels).item()
                
                predicted = (outputs > 0.5).float()  # Using fixed threshold of 0.5
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        val_accuracy = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        
        tracker.update(
            avg_train_loss, 
            avg_val_loss, 
            val_accuracy, 
            current_lr,
            0.5  # Fixed threshold
        )
        tracker.plot_training_progress()
        
        print(f'\nEpoch {epoch+1} Results:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print(f'Learning Rate: {current_lr}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
            no_improve = 0
            print("Saved new best model!")
        else:
            no_improve += 1
            if no_improve == patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
    
    # Feature importance analysis
    importances = analyze_feature_importance(model, val_loader, device)
    plot_feature_importances(importances)
    
    print("\nFeature Importance Analysis:")
    for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Final evaluation
    print("\nGenerating final evaluation metrics...")
    val_preds_binary = np.array(val_predictions) > 0.5
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(val_true_labels, val_preds_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(val_true_labels, val_predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    print("\nClassification Report:")
    print(classification_report(val_true_labels, val_preds_binary))
    
    return model, {
        'train_losses': tracker.train_losses,
        'val_losses': tracker.val_losses,
        'val_accuracies': tracker.val_accuracies,
        'best_val_loss': best_val_loss,
        'final_val_accuracy': val_accuracy,
        'feature_importances': importances
    }