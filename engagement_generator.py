import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import random
import os

class DayOfWeek(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

class CommentPatterns:
    def __init__(self):
        self.bot_patterns = {
            'comment_timing': {
                'burst_intervals': (5, 30),  # Seconds between comments in a burst
                'burst_size': (5, 15)        # Number of comments in a burst
            },
            'comment_text': {
                'repetitive_phrases': True,
                'comment_length': (5, 15)     # Typically shorter comments
            }
        }
        
        self.organic_patterns = {
            'comment_timing': {
                'natural_intervals': (30, 3600),  # More natural timing between comments
            },
            'comment_text': {
                'comment_length': (10, 50),        # More variable comment lengths
                'burst_probability': 0.1           # Probability of natural comment bursts
            }
        }

    def generate_comment_timeline(self, is_bot, num_comments, start_time):
        """Generate a timeline of comments with timestamps and basic metrics"""
        comments = []
        current_time = start_time
        
        if is_bot:
            # Generate bot-like comment patterns
            while len(comments) < num_comments:
                # Create comment bursts
                burst_size = random.randint(*self.bot_patterns['comment_timing']['burst_size'])
                for _ in range(min(burst_size, num_comments - len(comments))):
                    interval = random.randint(*self.bot_patterns['comment_timing']['burst_intervals'])
                    comment_length = random.randint(*self.bot_patterns['comment_text']['comment_length'])
                    
                    comments.append({
                        'timestamp': current_time,
                        'length': comment_length,
                        'is_duplicate_pattern': random.random() < 0.7
                    })
                    current_time += timedelta(seconds=interval)
                
                # Add larger gap between bursts
                current_time += timedelta(minutes=random.randint(5, 30))
        else:
            # Generate organic comment patterns
            while len(comments) < num_comments:
                interval = random.randint(*self.organic_patterns['comment_timing']['natural_intervals'])
                comment_length = random.randint(*self.organic_patterns['comment_text']['comment_length'])
                
                comments.append({
                    'timestamp': current_time,
                    'length': comment_length,
                    'is_duplicate_pattern': random.random() < 0.1
                })
                current_time += timedelta(seconds=interval)
                
                # Occasional natural bursts
                if random.random() < self.organic_patterns['comment_text']['burst_probability']:
                    # Create a natural burst of comments
                    burst_size = random.randint(2, 5)  # Smaller bursts for organic
                    for _ in range(min(burst_size, num_comments - len(comments))):
                        comments.append({
                            'timestamp': current_time,
                            'length': random.randint(*self.organic_patterns['comment_text']['comment_length']),
                            'is_duplicate_pattern': False
                        })
                        # More natural timing between burst comments
                        current_time += timedelta(seconds=random.randint(10, 60))
        
        return comments

class UserProfileMetrics:
    def generate_profile_metrics(self, is_bot):
        """Generate realistic profile metrics for engaging accounts"""
        if is_bot:
            return {
                'follower_following_ratio': random.uniform(0.1, 0.5),  # Bots often follow many but have few followers
                'avg_post_frequency': random.uniform(15, 30),          # Posts per day, unusually high for bots
                'profile_completion_score': random.uniform(0.3, 0.7)   # Often incomplete profiles
            }
        else:
            return {
                'follower_following_ratio': random.uniform(0.5, 2.0),  # More balanced ratio for real users
                'avg_post_frequency': random.uniform(1, 10),           # More natural posting frequency
                'profile_completion_score': random.uniform(0.7, 1.0)   # More complete profiles
            }

class BotEngagementPatterns:
    def __init__(self):
        self.bot_patterns = {
            'timing': {
                'instant_engagement': True,
                'time_variance': (0, 120),  # Increased variance to 2 hours
                'delayed_start': (0, 300)   # Random delay before bot activity
            },
            'engagement_spikes': {
                'sudden_increase': (2, 15),    # More variable spike intensity
                'duration_hours': (1, 6),      # Longer possible spike durations
                'gradual_increase': True,      # Sometimes ramp up instead of sudden spike
                'multiple_spikes': True        # Allow multiple engagement spikes
            },
            'engagement_ratios': {
                'like_to_view': (0.02, 0.45),  # More variable ratios
                'share_to_like': (0.01, 0.20),
                'natural_variance': 0.3        # Add natural variations
            }
        }

    def generate_bot_engagement(self, base_views, duration_hours=48):
        """Enhanced bot engagement generation with more natural patterns"""
        timeline = []
        
        # Determine bot behavior type
        behavior = random.choice(['aggressive', 'subtle', 'mixed'])
        
        # Create variable spike patterns
        spike_points = []
        if behavior == 'aggressive':
            num_spikes = np.random.randint(2, 4)
        elif behavior == 'subtle':
            num_spikes = np.random.randint(1, 2)
        else:
            num_spikes = np.random.randint(1, 3)
            
        for _ in range(num_spikes):
            spike_points.append({
                'hour': np.random.randint(0, duration_hours),
                'duration': np.random.randint(*self.bot_patterns['engagement_spikes']['duration_hours']),
                'multiplier': np.random.uniform(*self.bot_patterns['engagement_spikes']['sudden_increase']),
                'gradual': random.random() < 0.4  # 40% chance of gradual increase
            })
            
        base_noise = np.random.normal(1, 0.2, duration_hours)  # Add base noise
        
        for hour in range(duration_hours):
            # Determine if current hour is in any spike period
            spike_influence = 0
            for spike in spike_points:
                if spike['hour'] <= hour < (spike['hour'] + spike['duration']):
                    if spike['gradual']:
                        progress = (hour - spike['hour']) / spike['duration']
                        spike_influence = max(spike_influence, spike['multiplier'] * progress)
                    else:
                        spike_influence = max(spike_influence, spike['multiplier'])
            
            # Calculate base engagement with noise
            base_engagement = base_views * base_noise[hour]
            
            # Apply spike influence if any
            views = int(base_engagement * (1 + spike_influence))
            
            # Add natural variations to engagement ratios
            like_ratio = np.random.uniform(*self.bot_patterns['engagement_ratios']['like_to_view'])
            share_ratio = np.random.uniform(*self.bot_patterns['engagement_ratios']['share_to_like'])
            
            # Add some random noise to ratios
            like_ratio *= np.random.uniform(0.7, 1.3)
            share_ratio *= np.random.uniform(0.7, 1.3)
            
            engagement = {
                'hour': hour,
                'datetime': None,  # Will be set later
                'views': views,
                'likes': int(views * like_ratio),
                'shares': int(views * like_ratio * share_ratio)
            }
            
            timeline.append(engagement)
        
        # Calculate velocities after timeline is complete
        for i in range(1, len(timeline)):
            prev = timeline[i-1]
            curr = timeline[i]
            
            view_velocity = ((curr['views'] - prev['views']) / max(prev['views'], 1))
            like_velocity = ((curr['likes'] - prev['likes']) / max(prev['likes'], 1))
            
            curr['view_velocity'] = view_velocity + np.random.normal(0, 0.1)  # Add noise
            curr['like_velocity'] = like_velocity + np.random.normal(0, 0.1)
            curr['like_to_view_ratio'] = curr['likes'] / max(curr['views'], 1)
        
        # Set initial velocities to 0
        timeline[0]['view_velocity'] = 0
        timeline[0]['like_velocity'] = 0
        timeline[0]['like_to_view_ratio'] = timeline[0]['likes'] / max(timeline[0]['views'], 1)
        
        return timeline

class TimeEngagementPatterns:
    def __init__(self):
        self.peak_times = {
            DayOfWeek.MONDAY: [(10, 4), (16, 8)],     # 10 AM, 4 PM
            DayOfWeek.TUESDAY: [(14, 8), (16, 8)],    # 2 PM, 4 PM
            DayOfWeek.WEDNESDAY: [(14, 8), (16, 8)],  # 2 PM, 4 PM
            DayOfWeek.THURSDAY: [(10, 8), (18, 8)],   # 10 AM, 6 PM
            DayOfWeek.FRIDAY: [(8, 8), (20, 8)],      # 8 AM, 8 PM
            DayOfWeek.SATURDAY: [(22, 8)],            # 10 PM
            DayOfWeek.SUNDAY: [(14, 8)]               # 2 PM
        }
        
        self.time_multipliers = {
            'peak': (0.7, 1.2),        # More variable peaks
            'high': (0.5, 0.8),        # More overlap between levels
            'medium': (0.3, 0.6),
            'low': (0.1, 0.4)
        }
        
        self.natural_variance = 0.2

class TikTokEngagementGenerator:
    def __init__(self):
        self.engagement_tiers = {
            'nano': {
                'followers': (0, 10000),
                'engagement_rate': {
                    'average': 1.4269,
                    'top_10': 0.1736,
                    'growth_rate': 0.0728
                }
            },
            'micro': {
                'followers': (10000, 50000),
                'engagement_rate': {
                    'average': 0.0838,
                    'top_10': 0.1222,
                    'growth_rate': 0.0928
                }
            },
            'medium': {
                'followers': (50000, 100000),
                'engagement_rate': {
                    'average': 0.0764,
                    'top_10': 0.1369,
                    'growth_rate': 0.0887
                }
            },
            'macro': {
                'followers': (100000, 500000),
                'engagement_rate': {
                    'average': 0.0643,
                    'top_10': 0.1124,
                    'growth_rate': 0.0793
                }
            },
            'mega': {
                'followers': (500000, 5000000),
                'engagement_rate': {
                    'average': 0.0456,
                    'top_10': 0.0918,
                    'growth_rate': 0.0632
                }
            }
        }
        
        self.time_patterns = TimeEngagementPatterns()
        self.bot_patterns = BotEngagementPatterns()
        self.comment_patterns = CommentPatterns()
        self.profile_metrics = UserProfileMetrics()

    def generate_organic_engagement(self, post_datetime, follower_count, duration_hours=48):
        """Enhanced organic engagement with more natural variations"""
        timeline = []
        tier = self._get_influencer_tier(follower_count)
        base_rate = self.engagement_tiers[tier]['engagement_rate']['average']
        
        # Add natural variations
        base_noise = np.random.normal(1, 0.3, duration_hours)  # Increased noise
        
        # Random viral chance
        viral_chance = random.random() < 0.1  # 10% chance of viral post
        viral_multiplier = random.uniform(3, 8) if viral_chance else 1
        
        for hour in range(duration_hours):
            current_time = post_datetime + timedelta(hours=hour)
            day_of_week = current_time.weekday()
            hour_of_day = current_time.hour
            
            # Get time-based multiplier with added randomness
            time_multiplier = self._get_time_multiplier(day_of_week, hour_of_day)
            time_multiplier *= np.random.uniform(0.8, 1.2)  # Add 20% variance
            
            # Calculate views with multiple factors
            base_views = follower_count * random.uniform(0.05, 0.4)  # More variable view rate
            views = int(base_views * time_multiplier * base_noise[hour] * viral_multiplier)
            
            # Add occasional random spikes (even organic content can spike)
            if random.random() < 0.05:  # 5% chance per hour
                views = int(views * random.uniform(1.5, 3.0))
            
            # Calculate engagement with natural variations
            likes_ratio = base_rate * np.random.uniform(0.7, 1.3)  # 30% variance
            shares_ratio = likes_ratio * np.random.uniform(0.05, 0.15)
            
            engagement = {
                'hour': hour,
                'datetime': current_time,
                'views': views,
                'likes': int(views * likes_ratio),
                'shares': int(views * shares_ratio)
            }
            
            timeline.append(engagement)
        
        # Calculate velocities with natural variations
        for i in range(1, len(timeline)):
            prev = timeline[i-1]
            curr = timeline[i]
            
            view_velocity = ((curr['views'] - prev['views']) / max(prev['views'], 1))
            like_velocity = ((curr['likes'] - prev['likes']) / max(prev['likes'], 1))
            
            # Add noise to velocities
            curr['view_velocity'] = view_velocity + np.random.normal(0, 0.15)
            curr['like_velocity'] = like_velocity + np.random.normal(0, 0.15)
            curr['like_to_view_ratio'] = curr['likes'] / max(curr['views'], 1)
        
        # Set initial velocities
        timeline[0]['view_velocity'] = 0
        timeline[0]['like_velocity'] = 0
        timeline[0]['like_to_view_ratio'] = timeline[0]['likes'] / max(timeline[0]['views'], 1)
        
        return timeline

    def generate_mixed_engagement(self, post_datetime, follower_count, bot_ratio=0.3, duration_hours=48):
        """Generate more natural mixed engagement with enhanced features"""
        organic = self.generate_organic_engagement(post_datetime, follower_count)
        bot = self.bot_patterns.generate_bot_engagement(follower_count)
        
        # Add variable mixing ratios over time
        mixed_timeline = []
        base_bot_ratio = bot_ratio
        
        for org, bot in zip(organic, bot):
            # Vary bot ratio over time
            current_bot_ratio = base_bot_ratio * np.random.uniform(0.7, 1.3)
            
            mixed = {
                'hour': org['hour'],
                'datetime': org['datetime'],
                'views': int(org['views'] * (1-current_bot_ratio) + bot['views'] * current_bot_ratio),
                'likes': int(org['likes'] * (1-current_bot_ratio) + bot['likes'] * current_bot_ratio),
                'shares': int(org['shares'] * (1-current_bot_ratio) + bot['shares'] * current_bot_ratio)
            }
            
            # Add velocities
            if len(mixed_timeline) > 0:
                prev = mixed_timeline[-1]
                view_velocity = (mixed['views'] - prev['views']) / max(prev['views'], 1)
                like_velocity = (mixed['likes'] - prev['likes']) / max(prev['likes'], 1)
                mixed['view_velocity'] = view_velocity + np.random.normal(0, 0.1)
                mixed['like_velocity'] = like_velocity + np.random.normal(0, 0.1)
            else:
                mixed['view_velocity'] = 0
                mixed['like_velocity'] = 0
            
            mixed['like_to_view_ratio'] = mixed['likes'] / max(mixed['views'], 1)
            mixed_timeline.append(mixed)
            
        return mixed_timeline

    def _get_time_multiplier(self, day_of_week, hour):
        """Get engagement multiplier based on time patterns"""
        day = DayOfWeek(day_of_week)
        peak_hours = self.time_patterns.peak_times.get(day, [])
        
        for peak_hour, peak_score in peak_hours:
            if hour == peak_hour:
                return np.random.uniform(*self.time_patterns.time_multipliers['peak'])
        
        if (8 <= hour <= 10 or 14 <= hour <= 18) and day_of_week < 5:  # Weekdays
            return np.random.uniform(*self.time_patterns.time_multipliers['high'])
        
        if 6 <= hour <= 8 or 21 <= hour <= 23:
            return np.random.uniform(*self.time_patterns.time_multipliers['medium'])
        
        if 0 <= hour <= 5:
            return np.random.uniform(*self.time_patterns.time_multipliers['low'])
            
        return np.random.uniform(*self.time_patterns.time_multipliers['medium'])

    def _get_influencer_tier(self, follower_count):
        """Determine influencer tier based on follower count"""
        for tier, data in self.engagement_tiers.items():
            min_followers, max_followers = data['followers']
            if min_followers <= follower_count < max_followers:
                return tier
        return 'mega'

def generate_dataset(num_posts=1000):
    """Generate a complete dataset with various engagement patterns and enhanced features"""
    generator = TikTokEngagementGenerator()
    dataset = []
    
    print("Generating synthetic dataset...")
    for i in range(num_posts):
        if i % 10 == 0:  # Show progress every 10 posts
            print(f"Progress: {i}/{num_posts} posts generated")

    for _ in range(num_posts):
        # Random posting time
        start_date = datetime(2024, 1, 1)
        random_days = random.randint(0, 60)
        random_hours = random.randint(0, 23)
        post_datetime = start_date + timedelta(days=random_days, hours=random_hours)
        
        # Random follower count
        follower_count = random.randint(1000, 1000000)
        
        # Decide engagement type with more mixed cases
        engagement_type = random.choices(
            ['organic', 'bot', 'mixed'],
            weights=[0.6, 0.2, 0.2]  # Adjusted weights
        )[0]
        
        if engagement_type == 'organic':
            raw_timeline = generator.generate_organic_engagement(post_datetime, follower_count)
            is_bot = False
            bot_ratio = 0.0
        elif engagement_type == 'bot':
            raw_timeline = generator.bot_patterns.generate_bot_engagement(follower_count)
            is_bot = True
            bot_ratio = 1.0
        else:
            bot_ratio = random.uniform(0.3, 0.7)
            raw_timeline = generator.generate_mixed_engagement(post_datetime, follower_count, bot_ratio)
            is_bot = True

        # Generate comment data
        num_comments = min(
            int(sum(hour['likes'] for hour in raw_timeline) * random.uniform(0.1, 0.3)),
            1000  # Cap maximum comments at 1000
        )
        comments = generator.comment_patterns.generate_comment_timeline(
            is_bot,
            num_comments,
            post_datetime
        )

        # Generate profile metrics
        profile_metrics = generator.profile_metrics.generate_profile_metrics(is_bot)
        
        dataset.append({
            'post_datetime': post_datetime,
            'follower_count': follower_count,
            'engagement_type': engagement_type,
            'is_bot': is_bot,
            'bot_ratio': bot_ratio,
            'timeline': raw_timeline,
            'comments': comments,
            'profile_metrics': profile_metrics
        })
    
    # Convert to pandas DataFrame for easier analysis
    processed_data = []
    for post in dataset:
        for hour_data in post['timeline']:
            record = {
                'post_datetime': post['post_datetime'],
                'follower_count': post['follower_count'],
                'engagement_type': post['engagement_type'],
                'is_bot': post['is_bot'],
                'bot_ratio': post['bot_ratio'],
                'hour': hour_data['hour'],
                'views': hour_data['views'],
                'likes': hour_data['likes'],
                'shares': hour_data['shares'],
                'view_velocity': hour_data.get('view_velocity', 0),
                'like_velocity': hour_data.get('like_velocity', 0),
                'like_to_view_ratio': hour_data.get('like_to_view_ratio', 0),
                # Add comment metrics
                'comment_burst_frequency': len([c for c in post['comments'] if c['is_duplicate_pattern']]) / max(len(post['comments']), 1),
                'avg_comment_length': sum(c['length'] for c in post['comments']) / max(len(post['comments']), 1),
                'comment_timing_variance': np.std([c['timestamp'].timestamp() for c in post['comments']]) if post['comments'] else 0,
                # Add profile metrics
                'follower_following_ratio': post['profile_metrics']['follower_following_ratio'],
                'avg_post_frequency': post['profile_metrics']['avg_post_frequency'],
                'profile_completion_score': post['profile_metrics']['profile_completion_score']
            }
            processed_data.append(record)
    
    df = pd.DataFrame(processed_data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/enhanced_engagement_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset generated and saved to {output_path}")
    print("\nSample of the data:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nEngagement type distribution:")
    print(df['engagement_type'].value_counts())
    
    return df

if __name__ == "__main__":
    df = generate_dataset(num_posts=1000)