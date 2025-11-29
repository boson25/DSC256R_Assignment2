"""
Data loading utilities for Amazon Reviews 2023

Uses official Hugging Face datasets library as recommended by:
https://amazon-reviews-2023.github.io/data_loading/index.html
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from collections import defaultdict
import pickle
import os


def download_amazon_reviews(category: str = "All_Beauty", 
                            data_dir: str = "../data/raw",
                            use_sample: bool = True,
                            max_reviews: int = 100000) -> str:
    """
    Download Amazon Reviews 2023 directly from UCSD repository.
    
    Downloads the raw .jsonl.gz file and processes it.
    Source: https://amazon-reviews-2023.github.io/
    
    Args:
        category: Product category
        data_dir: Directory to save data
        use_sample: If True, load only max_reviews rows
        max_reviews: Maximum number of reviews in sample mode
    
    Returns:
        Path to saved parquet file
    """
    import gzip
    import json
    import requests
    from tqdm import tqdm
    
    print(f"Source: UCSD McAuley Lab Repository")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # https://amazon-reviews-2023.github.io/main.html
    base_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2023/raw/review_categories"
    file_url = f"{base_url}/{category}.jsonl.gz"
    
    gz_file = os.path.join(data_dir, f"{category}.jsonl.gz")
    
    # Download the file
    if os.path.exists(gz_file):
        print(f"File already exists: {gz_file}")
        print("Using existing file. Delete it to re-download.")
    else:
        print(f"Downloading from: {file_url}")
        
        try:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(gz_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=category) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f" Downloaded successfully!")
            
        except requests.exceptions.RequestException as e:
            print(f" Download failed: {e}")
            print(f"\n Please try:")
            print(f"   1. Check if this URL works in browser: {file_url}")
            print(f"   2. Try a different category")
            print(f"   3. Download manually and save to: {gz_file}")
            raise
    
    # Load and parse the gzipped JSON file
    reviews = []
    
    with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
        if use_sample:
            pbar = tqdm(total=max_reviews, desc="Loading reviews")
            
            for i, line in enumerate(f):
                if i >= max_reviews:
                    break
                try:
                    review = json.loads(line)
                    reviews.append(review)
                    pbar.update(1)
                except json.JSONDecodeError:
                    continue
            pbar.close()
        else:
            for line in tqdm(f, desc="Loading reviews"):
                try:
                    review = json.loads(line)
                    reviews.append(review)
                except json.JSONDecodeError:
                    continue
    
    # Convert to DataFrame
    df = pd.DataFrame(reviews)
    
    # Save as parquet
    output_path = os.path.join(data_dir, f"{category}_reviews.parquet")
    df.to_parquet(output_path, index=False)
    
    print(f"\n Loaded {len(df):,} reviews")
    print(f" Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def load_reviews_from_parquet(filepath: str) -> pd.DataFrame:
    """Load reviews from saved parquet file."""
    print(f"Loading reviews from: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} reviews")
    return df


def preprocess_reviews(df: pd.DataFrame,
                       min_user_reviews: int = 5,
                       min_item_reviews: int = 5) -> pd.DataFrame:
    """
    Preprocess Amazon reviews data.
    
    Steps:
    1. Remove missing ratings
    2. K-core filtering (iterative user/item filtering)
    """
    print("\n" + "="*60)
    print("Processing data")
    print("="*60)
    
    original_size = len(df)
    print(f"\nOriginal dataset: {original_size:,} reviews")
    
    # Rename columns for consistency
    column_mapping = {
        'parent_asin': 'item_id',
        'user_id': 'user_id',
        'rating': 'rating',
        'text': 'text',
        'timestamp': 'timestamp'
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Remove missing ratings
    df = df[df['rating'].notna() & (df['rating'] >= 1.0) & (df['rating'] <= 5.0)]
    print(f"After removing invalid ratings (0 or >5): {len(df):,} reviews ({len(df)/original_size*100:.1f}%)")

    
    # K-core filtering
    print(f"\nApplying k-core filtering (min_user={min_user_reviews}, min_item={min_item_reviews})...")
    
    iteration = 0
    while True:
        iteration += 1
        prev_size = len(df)
        
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_user_reviews].index
        valid_items = item_counts[item_counts >= min_item_reviews].index
        
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        print(f"  Iteration {iteration}: {len(df):,} reviews (removed {prev_size - len(df):,})")
        
        if len(df) == prev_size:
            break
    
    print(f"\nAfter k-core: {len(df):,} reviews ({len(df)/original_size*100:.1f}% retained)")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique items: {df['item_id'].nunique():,}")
    
    return df.reset_index(drop=True)


def create_user_item_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create integer index mappings for users and items."""
    print("\n" + "="*60)
    print("CREATING ID MAPPINGS")
    print("="*60)
    
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
    
    print(f"\nCreated mappings:")
    print(f"  Users: {len(user_to_idx):,}")
    print(f"  Items: {len(item_to_idx):,}")
    
    return user_to_idx, item_to_idx


def create_train_test_split(df: pd.DataFrame,
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test split."""
    print("\n" + "="*60)
    print("CREATING TRAIN/TEST SPLIT")
    print("="*60)
    
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * (1 - test_size))
    
    train_df = df_shuffled[:split_idx].reset_index(drop=True)
    test_df = df_shuffled[split_idx:].reset_index(drop=True)
    
    print(f"\nSplit: {100*(1-test_size):.0f}% train / {100*test_size:.0f}% test")
    print(f"  Train: {len(train_df):,} reviews")
    print(f"  Test:  {len(test_df):,} reviews")
    
    return train_df, test_df


def save_processed_data(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       user_to_idx: Dict[str, int],
                       item_to_idx: Dict[str, int],
                       output_dir: str = "../data/processed") -> None:
    """Save all processed data files."""
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving to: {output_dir}")
    
    # Save DataFrames
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    test_df.to_parquet(f"{output_dir}/test.parquet", index=False)
    
    # Save mappings
    with open(f"{output_dir}/user_to_idx.pkl", 'wb') as f:
        pickle.dump(user_to_idx, f)
    
    with open(f"{output_dir}/item_to_idx.pkl", 'wb') as f:
        pickle.dump(item_to_idx, f)
    
    # Save metadata
    metadata = {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_users': len(user_to_idx),
        'n_items': len(item_to_idx),
        'rating_mean': float(train_df['rating'].mean()),
        'rating_std': float(train_df['rating'].std()),
    }
    
    with open(f"{output_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Saved files:")
    print(f"  - train.csv & train.parquet")
    print(f"  - test.csv & test.parquet")
    print(f"  - user_to_idx.pkl ({len(user_to_idx):,} users)")
    print(f"  - item_to_idx.pkl ({len(item_to_idx):,} items)")
    print(f"  - metadata.pkl")


def load_splits(data_dir: str = "../data/processed",
               use_parquet: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Load preprocessed train/test splits.
    
    """
  
    
    if use_parquet:
        train_df = pd.read_parquet(f"{data_dir}/train.parquet")
        test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    else:
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")
    
    with open(f"{data_dir}/user_to_idx.pkl", 'rb') as f:
        user_to_idx = pickle.load(f)
    
    with open(f"{data_dir}/item_to_idx.pkl", 'rb') as f:
        item_to_idx = pickle.load(f)
    
    print(f" Loaded:")
    print(f"  Train: {len(train_df):,} reviews")
    print(f"  Test:  {len(test_df):,} reviews")
    print(f"  Users: {len(user_to_idx):,}")
    print(f"  Items: {len(item_to_idx):,}")
    
    return train_df, test_df, user_to_idx, item_to_idx


def get_dataset_statistics(train_df: pd.DataFrame, 
                          test_df: pd.DataFrame,
                          user_to_idx: Dict,
                          item_to_idx: Dict) -> Dict:
    """Compute comprehensive dataset statistics."""
    full_df = pd.concat([train_df, test_df])
    
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    n_reviews = len(full_df)
    
    stats = {
    'n_reviews_total': n_reviews,
    'n_reviews_train': len(train_df),
    'n_reviews_test': len(test_df),
    'n_users': n_users,
    'n_items': n_items,
    'density': n_reviews / (n_users * n_items),
    'sparsity': 1 - (n_reviews / (n_users * n_items)),
    'rating_mean': full_df['rating'].mean(),
    'rating_std': full_df['rating'].std(),
    'rating_min': full_df['rating'].min(),
    'rating_max': full_df['rating'].max(),
    'rating_distribution': full_df['rating'].value_counts().sort_index().to_dict(),
    'reviews_per_user_mean': full_df['user_id'].value_counts().mean(),
    'reviews_per_user_median': full_df['user_id'].value_counts().median(),
    'reviews_per_user_max': full_df['user_id'].value_counts().max(),
    'reviews_per_item_mean': full_df['item_id'].value_counts().mean(),
    'reviews_per_item_median': full_df['item_id'].value_counts().median(),
    'reviews_per_item_max': full_df['item_id'].value_counts().max(),
}

    return stats