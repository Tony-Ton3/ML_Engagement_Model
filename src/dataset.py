import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EngagementDataset(Dataset):
    def __init__(self, df):
        # Group features by type for better organization
        engagement_features = [
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
        
        comment_features = [
            'comment_burst_frequency',
            'avg_comment_length',
            'comment_timing_variance'
        ]
        
        profile_features = [
            'follower_following_ratio',
            'avg_post_frequency',
            'profile_completion_score'
        ]
        
        # Combine all features
        features = engagement_features + comment_features + profile_features
        
        # Convert to numpy for preprocessing
        self.features = df[features].values
        
        # Handle infinite values and NaN
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Robust normalization with simpler approach
        for i in range(self.features.shape[1]):
            col = self.features[:, i]
            col_mean = np.mean(col)
            col_std = np.std(col)
            if col_std != 0:
                self.features[:, i] = (col - col_mean) / col_std
            else:
                self.features[:, i] = col - col_mean
        
        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(df['is_bot'].astype(float).values)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]