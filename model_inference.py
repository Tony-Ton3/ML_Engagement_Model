import torch
import pandas as pd
import numpy as np
from src.model import EngagementClassifier
from src.dataset import EngagementDataset

def predict_engagement(sample_data):
    # Load the trained model
    model = EngagementClassifier()
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    model.eval()
    
    # Prepare the data like we did in training
    features = [
        'follower_count',
        'bot_ratio',
        'hour',
        'views',
        'likes',
        'shares',
        'view_velocity',
        'like_velocity',
        'like_to_view_ratio'
    ]
    
    # Convert to DataFrame if it's a dictionary
    if isinstance(sample_data, dict):
        sample_data = pd.DataFrame([sample_data])
    
    # Extract features and normalize them
    X = sample_data[features].values
    
    # Convert to tensor
    X = torch.FloatTensor(X)
    
    # Get prediction
    with torch.no_grad():
        output = model(X)
        prediction = (output >= 0.5).float()
        probability = output.item()
    
    return "Bot" if prediction.item() == 1 else "Organic", probability

# Example usage
sample_engagement = {
    'follower_count': 100000,
    'bot_ratio': 0,
    'hour': 12,
    'views': 5000,
    'likes': 500,
    'shares': 50,
    'view_velocity': 0.2,
    'like_velocity': 0.15,
    'like_to_view_ratio': 0.1
}

result, prob = predict_engagement(sample_engagement)
print(f"Prediction: {result} (confidence: {prob:.2%})")

# Let's try some different patterns
samples = [
    # Normal organic engagement
    {
        'follower_count': 100000,
        'bot_ratio': 0,
        'hour': 12,
        'views': 5000,
        'likes': 500,
        'shares': 50,
        'view_velocity': 0.2,
        'like_velocity': 0.15,
        'like_to_view_ratio': 0.1
    },
    # Suspicious spike in engagement
    {
        'follower_count': 100000,
        'bot_ratio': 0,
        'hour': 1,
        'views': 50000,
        'likes': 25000,
        'shares': 5000,
        'view_velocity': 5.0,
        'like_velocity': 10.0,
        'like_to_view_ratio': 0.5
    },
    # Mixed engagement pattern
    {
        'follower_count': 100000,
        'bot_ratio': 0.3,
        'hour': 24,
        'views': 15000,
        'likes': 3000,
        'shares': 300,
        'view_velocity': 1.5,
        'like_velocity': 2.0,
        'like_to_view_ratio': 0.2
    }
]

print("\nTesting multiple patterns:")
for i, sample in enumerate(samples, 1):
    result, prob = predict_engagement(sample)
    print(f"\nPattern {i}:")
    print(f"Engagement metrics: {sample}")
    print(f"Prediction: {result} (confidence: {prob:.2%})")