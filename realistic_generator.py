import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

class PostContext:
    def __init__(self):
        self.audio_trends = {
            'viral': (0.7, 1.0),    # Viral audio trend strength range
            'trending': (0.4, 0.7),  # Regular trending audio range
            'normal': (0.1, 0.4)     # Non-trending audio range
        }
        
        self.content_types = {
            'challenge': (0.6, 0.9),     # Trending challenge engagement boost
            'trend_follow': (0.4, 0.7),  # Following current trend
            'original': (0.2, 0.5)       # Original content
        }

    def generate_context(self):
        # Determine audio status
        audio_type = random.choices(
            ['viral', 'trending', 'normal'],
            weights=[0.1, 0.2, 0.7]
        )[0]
        audio_strength = random.uniform(*self.audio_trends[audio_type])
        
        # Determine content type
        content_type = random.choices(
            ['challenge', 'trend_follow', 'original'],
            weights=[0.15, 0.25, 0.6]
        )[0]
        content_strength = random.uniform(*self.content_types[content_type])
        
        # Generate video metrics
        video_length = random.randint(15, 180)  # 15s to 3min
        
        # Calculate base watch percentage based on content quality
        base_watch = 0.3  # Base 30% watch retention
        if audio_type in ['viral', 'trending']:
            base_watch += 0.2 * audio_strength
        if content_type in ['challenge', 'trend_follow']:
            base_watch += 0.15 * content_strength
            
        # Add some noise to watch time
        watch_percentage = min(0.95, max(0.15, base_watch + random.uniform(-0.1, 0.1)))
        
        return {
            'audio_trending': audio_type != 'normal',
            'audio_strength': audio_strength,
            'content_trending': content_type != 'original',
            'content_strength': content_strength,
            'video_length': video_length,
            'avg_watch_percentage': watch_percentage
        }

class EngagementGenerator:
    def __init__(self):
        self.post_context = PostContext()
        
        # Base engagement patterns by account size
        self.engagement_patterns = {
            'small': {  # <10k followers
                'views': (1000, 10000),
                'like_rate': (0.15, 0.40),
                'comment_rate': (0.01, 0.05),
                'share_rate': (0.01, 0.03)
            },
            'medium': {  # 10k-100k followers
                'views': (8000, 50000),
                'like_rate': (0.10, 0.30),
                'comment_rate': (0.005, 0.03),
                'share_rate': (0.005, 0.02)
            },
            'large': {  # >100k followers
                'views': (40000, 500000),
                'like_rate': (0.05, 0.20),
                'comment_rate': (0.002, 0.02),
                'share_rate': (0.002, 0.01)
            }
        }
        
        # Comment timing patterns
        self.comment_patterns = {
            'organic': {
                'interval_range': (30, 3600),    # 30s to 1h between comments
                'burst_chance': 0.2              # Natural comment burst chance
            },
            'bot': {
                'interval_range': (1, 10),       # 1-10s between comments
                'burst_size': (5, 15)            # Comments per burst
            }
        }

    def calculate_bot_ratio(self, context):
        """Calculate probable bot engagement ratio based on content context"""
        base_ratio = 0.3  # Base assumption of 30% bot engagement
        
        # Reduce bot ratio for high-quality indicators
        if context['avg_watch_percentage'] > 0.6:
            base_ratio *= 0.6  # High watch time suggests real engagement
        
        if context['audio_trending']:
            base_ratio *= (1 - context['audio_strength'] * 0.4)
            
        if context['content_trending']:
            base_ratio *= (1 - context['content_strength'] * 0.3)
        
        # Increase bot ratio for low-quality indicators
        if context['avg_watch_percentage'] < 0.3:
            base_ratio *= 1.5  # Low watch time suggests potential fake engagement
            
        if not context['audio_trending'] and not context['content_trending']:
            base_ratio *= 1.3  # Non-trending content more likely to use bots
            
        # Add some randomness
        noise = random.uniform(-0.1, 0.1)
        final_ratio = max(0.05, min(0.95, base_ratio + noise))
        
        return final_ratio

    def generate_engagement(self, post_datetime, follower_count, duration_hours=24):
        # Generate post context
        context = self.post_context.generate_context()
        
        # Calculate bot ratio based on context
        bot_ratio = self.calculate_bot_ratio(context)
        
        # Determine account tier
        if follower_count < 10000:
            tier = 'small'
        elif follower_count < 100000:
            tier = 'medium'
        else:
            tier = 'large'
        
        patterns = self.engagement_patterns[tier]
        
        # Generate base metrics
        base_views = random.randint(*patterns['views'])
        timeline = []
        
        for hour in range(duration_hours):
            # Calculate hour's engagement
            view_multiplier = 1 + (bot_ratio * random.uniform(0.5, 1.5))
            views = int(base_views * view_multiplier)
            
            # Calculate engagement rates
            like_rate = random.uniform(*patterns['like_rate'])
            comment_rate = random.uniform(*patterns['comment_rate'])
            share_rate = random.uniform(*patterns['share_rate'])
            
            # Adjust rates based on context
            if context['audio_trending'] or context['content_trending']:
                like_rate *= (1 + context['content_strength'] * 0.3)
                share_rate *= (1 + context['content_strength'] * 0.2)
            
            engagement = {
                'hour': hour,
                'datetime': post_datetime + timedelta(hours=hour),
                'views': views,
                'likes': int(views * like_rate),
                'comments': int(views * comment_rate),
                'shares': int(views * share_rate)
            }
            
            # Calculate velocities
            if timeline:
                prev = timeline[-1]
                engagement['view_velocity'] = (engagement['views'] - prev['views']) / max(prev['views'], 1)
                engagement['like_velocity'] = (engagement['likes'] - prev['likes']) / max(prev['likes'], 1)
            else:
                engagement['view_velocity'] = 0
                engagement['like_velocity'] = 0
            
            timeline.append(engagement)
        
        return timeline, context, bot_ratio

def generate_dataset(num_posts=1000):
    generator = EngagementGenerator()
    dataset = []
    
    print("Generating synthetic dataset...")
    for i in range(num_posts):
        if i % 10 == 0:
            print(f"Progress: {i}/{num_posts} posts generated")
        
        # Generate post parameters
        post_datetime = datetime(2024, 1, 1) + timedelta(
            days=random.randint(0, 60),
            hours=random.randint(0, 23)
        )
        follower_count = random.randint(1000, 1000000)
        
        # Generate engagement data
        timeline, context, bot_ratio = generator.generate_engagement(
            post_datetime,
            follower_count
        )
        
        # Store data
        for hour_data in timeline:
            record = {
                'post_datetime': post_datetime,
                'follower_count': follower_count,
                'bot_ratio': bot_ratio,
                'avg_watch_percentage': context['avg_watch_percentage'],
                'audio_trending': int(context['audio_trending']),
                'audio_strength': context['audio_strength'],
                'content_trending': int(context['content_trending']),
                'content_strength': context['content_strength'],
                'video_length': context['video_length'],
                **hour_data
            }
            dataset.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Calculate additional ratios
    df['like_to_view_ratio'] = df['likes'] / df['views'].replace(0, 1)
    df['comment_to_view_ratio'] = df['comments'] / df['views'].replace(0, 1)
    df['share_to_view_ratio'] = df['shares'] / df['views'].replace(0, 1)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    output_path = 'data/tiktok_engagement_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset generated and saved to {output_path}")
    print("\nSample of the data:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    
    return df

if __name__ == "__main__":
    df = generate_dataset(num_posts=10000)