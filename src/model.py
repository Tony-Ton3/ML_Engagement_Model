import torch
import torch.nn as nn
import torch.nn.functional as F

class EngagementClassifier(nn.Module):
    def __init__(self, input_features=15):  # Updated for new features
        super().__init__()
        
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Second layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.network(x)

# Simplified loss function
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return self.bce(pred, target)

def analyze_feature_importance(model, val_loader, device):
    feature_names = [
        # Engagement features
        'follower_count', 'bot_ratio', 'hour', 'views', 'likes',
        'shares', 'view_velocity', 'like_velocity', 'like_to_view_ratio',
        # Comment features
        'comment_burst_frequency', 'avg_comment_length', 'comment_timing_variance',
        # Profile features
        'follower_following_ratio', 'avg_post_frequency', 'profile_completion_score'
    ]
    
    feature_importance_dict = {name: 0.0 for name in feature_names}
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for features, _ in val_loader:
            features = features.to(device)
            baseline = model(features)
            
            for i in range(features.shape[1]):
                perturbed = features.clone()
                perturbed[:, i] += torch.randn_like(perturbed[:, i]) * 0.1
                delta = (model(perturbed) - baseline).abs().mean().item()
                feature_importance_dict[feature_names[i]] += delta
            
            total_samples += 1
    
    # Average the importances
    for feature in feature_importance_dict:
        feature_importance_dict[feature] /= total_samples
        
    return feature_importance_dict