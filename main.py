import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.dataset import EngagementDataset
from src.model import EngagementClassifier
from src.train import train_model, create_balanced_sampler

def print_dataset_stats(df, name):
    """Print statistics about the dataset split"""
    total = len(df)
    bot_count = df['is_bot'].sum()
    organic_count = total - bot_count
    
    print(f"\n{name} Dataset Statistics:")
    print(f"Total samples: {total}")
    print(f"Bot samples: {bot_count} ({(bot_count/total)*100:.1f}%)")
    print(f"Organic samples: {organic_count} ({(organic_count/total)*100:.1f}%)")
    
    # Print feature statistics
    print(f"\nFeature ranges:")
    features_to_check = [
        # Engagement metrics
        'views', 'likes', 'shares', 'view_velocity', 'like_velocity', 
        # Comment metrics
        'comment_burst_frequency', 'avg_comment_length', 'comment_timing_variance',
        # Profile metrics
        'follower_following_ratio', 'avg_post_frequency', 'profile_completion_score'
    ]
    
    for feature in features_to_check:
        print(f"{feature}: {df[feature].mean():.2f} ± {df[feature].std():.2f}")

def main():
    print("Loading dataset...")
    df = pd.read_csv('data/enhanced_engagement_dataset.csv')
    
    # Remove any infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    print("Splitting data...")
    # Stratified split to maintain class distribution
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['is_bot'],
        random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.2,
        stratify=train_df['is_bot'],
        random_state=42
    )
    
    # Print statistics for each split
    print_dataset_stats(train_df, "Training")
    print_dataset_stats(val_df, "Validation")
    print_dataset_stats(test_df, "Test")
    
    print("\nCreating datasets...")
    # Create datasets with the new feature set
    train_dataset = EngagementDataset(train_df)
    val_dataset = EngagementDataset(val_df)
    test_dataset = EngagementDataset(test_df)
    
    # Create balanced sampler for training
    train_sampler = create_balanced_sampler(train_dataset)
    
    print("Setting up data loaders...")
    # Create dataloaders with slightly larger batch size for efficiency
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,  # Increased batch size
        sampler=train_sampler,
        num_workers=4,  # Added parallel data loading
        pin_memory=True  # Improved GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64,
        num_workers=4,
        pin_memory=True
    )
    
    print("\nStarting model training...")
    try:
        model, stats = train_model(train_loader, val_loader)
        
        print("\nTraining completed successfully!")
        print(f"Best validation loss: {stats['best_val_loss']:.4f}")
        print(f"Final validation accuracy: {stats['final_val_accuracy']:.2f}%")
        
        print("\nFeature Importance Summary:")
        for feature, importance in sorted(
            stats['feature_importances'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]:
            print(f"{feature}: {importance:.4f}")
            
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()