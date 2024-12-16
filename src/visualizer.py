import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import r2_score
import pandas as pd

class ModelVisualizer:
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def plot_training_history(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.close()

    def plot_prediction_vs_actual(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy()
            actual = y_test.cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predictions, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
        plt.xlabel('Actual Bot Ratio')
        plt.ylabel('Predicted Bot Ratio')
        plt.title('Predicted vs Actual Bot Ratio')
        plt.grid(True)
        plt.savefig('prediction_vs_actual.png')
        plt.close()

    def plot_error_distribution(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy()
            actual = y_test.cpu().numpy()
            errors = predictions - actual

        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig('error_distribution.png')
        plt.close()

    def plot_feature_importance(self, df):
        # Get feature names
        features = self.trainer.prepare_features(df).columns.tolist()
        
        # Create synthetic variations to measure impact
        feature_importance = []
        X = self.trainer.prepare_features(df)
        X_scaled = self.trainer.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        base_prediction = self.model(X_tensor).mean().item()
        
        for i, feature in enumerate(features):
            X_modified = X_scaled.copy()
            X_modified[:, i] += 1  # Increase feature by 1 std
            X_modified_tensor = torch.FloatTensor(X_modified).to(self.device)
            new_prediction = self.model(X_modified_tensor).mean().item()
            importance = abs(new_prediction - base_prediction)
            feature_importance.append((feature, importance))
        
        # Sort and plot
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        features, importance = zip(*feature_importance)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(importance), y=list(features))
        plt.title('Feature Importance')
        plt.xlabel('Absolute Impact on Prediction')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

# Usage example in main.py
def create_visualizations(model, trainer, df):
    visualizer = ModelVisualizer(model, trainer)
    
    # Get the data splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.prepare_data(df)
    
    # Generate all plots
    visualizer.plot_prediction_vs_actual(X_test, y_test)
    visualizer.plot_error_distribution(X_test, y_test)
    visualizer.plot_feature_importance(df)
    
    return "Visualizations have been saved as PNG files in the current directory."