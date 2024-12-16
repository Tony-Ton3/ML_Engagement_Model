import pandas as pd
from src.train import BotEngagementTrainer
from src.visualizer import create_visualizations

def main():
    # Load data
    df = pd.read_csv('data/tiktok_engagement_dataset.csv')
    
    # Initialize trainer
    trainer = BotEngagementTrainer(learning_rate=0.001)
    
    # Train model
    model = trainer.train_model(df, epochs=100)
    
    # Get data splits
    (_, _), (_, _), (X_test, y_test) = trainer.prepare_data(df)
    
    # Evaluate model
    metrics = trainer.evaluate_model(model, X_test, y_test)
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

        # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(model, trainer, df)

if __name__ == "__main__":
    main()