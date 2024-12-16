import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.model import EngagementPredictor

class BotEngagementTrainer:
    def __init__(self, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.learning_rate = learning_rate
        
    def prepare_features(self, df):

        features = pd.DataFrame()

        # Engagement ratios
        features['like_to_view_ratio'] = df['like_to_view_ratio']
        features['comment_to_view_ratio'] = df['comment_to_view_ratio']
        features['share_to_view_ratio'] = df['share_to_view_ratio']
    
        # Velocity metrics
        features['view_velocity'] = df['view_velocity']
        features['like_velocity'] = df['like_velocity']
    
        # Content quality indicators
        features['audio_strength'] = df['audio_strength']
        features['content_strength'] = df['content_strength']
        features['avg_watch_percentage'] = df['avg_watch_percentage']
    
        # Video metadata
        features['video_length'] = df['video_length']
        features['follower_count'] = df['follower_count']
    
        # Trending indicators
        features['audio_trending'] = df['audio_trending']
        features['content_trending'] = df['content_trending']
    
        # Time features
        features['hour'] = df['hour']
    
        return features
    
    def prepare_data(self, df):
        """Prepare features and split data"""
        X = self.prepare_features(df)
        y = df['bot_ratio']
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1).to(self.device)
        
        return (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), (X_test_tensor, y_test_tensor)
    
    def train_model(self, df, epochs=100, batch_size=32):
        """Train the model on the provided data"""
        # Prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(df)
        
        # Initialize model
        input_size = X_train.shape[1]  # Number of features
        model = EngagementPredictor(input_size=input_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10  # Early stopping patience
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Input size: {input_size} features")
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train):.4f}, Val Loss: {val_loss:.4f}')
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping after {epoch + 1} epochs')
                        break
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Calculate evaluation metrics"""
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            mse = nn.MSELoss()(predictions, y_test).item()
            mae = nn.L1Loss()(predictions, y_test).item()
            rmse = np.sqrt(mse)
            
            # R-squared calculation
            ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
            ss_res = ((y_test - predictions) ** 2).sum().item()
            r2 = 1 - (ss_res / ss_tot)
            
            # Additional metrics
            predictions_np = predictions.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_test_np - predictions_np) / y_test_np)) * 100
            
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

# For testing the trainer directly
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('data/tiktok_engagement_dataset.csv')
    
    # Initialize and train model
    trainer = BotEngagementTrainer(learning_rate=0.001)
    model = trainer.train_model(df, epochs=100)
    
    # Get data splits
    (_, _), (_, _), (X_test, y_test) = trainer.prepare_data(df)
    
    # Evaluate model
    metrics = trainer.evaluate_model(model, X_test, y_test)
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")