# Dataset Documentation

## Amazon Reviews 2023 - All_Beauty Category

**Source:** [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

---

## Quick Start for Team
```python
import sys
sys.path.append('src')
from data_loader import load_splits

# Load preprocessed data
train_df, test_df, user_to_idx, item_to_idx = load_splits()


```

---

## Dataset Statistics

- **Total Reviews:** 10,245
- **Train Reviews:** 8,196 (80%)
- **Test Reviews:** 2,049 (20%)
- **Unique Users:** 3,543
- **Unique Items:** 2,998
- **Sparsity:** 99.90%
- **Average Rating:** 4.21
- **Rating Range:** 1.0 - 5.0

---

## File Structure
```
data/
├── raw/
│   ├── All_Beauty.jsonl              # Original downloaded data (300K reviews)
│   └── All_Beauty_reviews.parquet    # Converted to parquet format
└── processed/
    ├── train.csv                      # Training set (CSV format)
    ├── train.parquet                  # Training set (Parquet format - faster)
    ├── test.csv                       # Test set (CSV format)
    ├── test.parquet                   # Test set (Parquet format - faster)
    ├── user_to_idx.pkl                # User ID → integer index mapping
    ├── item_to_idx.pkl                # Item ID → integer index mapping
    └── metadata.pkl                   # Dataset statistics
```

---

## Data Schema

### Review Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `user_id` | str | Unique user identifier | "A2SUAM1J3GNN3B" |
| `item_id` | str | Unique item identifier (parent_asin) | "B00006WNMB" |
| `rating` | float | Rating from 1.0 to 5.0 | 5.0 |
| `text` | str | Review text content | "Great product!" |
| `timestamp` | int | Unix timestamp in milliseconds | 1577836800000 |

### ID Mappings

- `user_to_idx`: Dictionary mapping user_id (string) → integer index [0, 3542]
- `item_to_idx`: Dictionary mapping item_id (string) → integer index [0, 2997]

**Why mappings?** Matrix factorization models need integer indices for embedding lookups.

---

## Preprocessing Applied

1. **Source:** Downloaded 300,000 reviews from Hugging Face
2. **Invalid Rating Removal:** Removed reviews with rating = 0 or missing ratings
3. **K-Core Filtering:**
   - Kept only users with ≥2 reviews
   - Kept only items with ≥2 reviews
   - Applied iteratively until convergence
   - Result: 10,245 reviews retained (3.4% of original 300K sample)
4. **Train/Test Split:** Random 80/20 split with seed=42 for reproducibility

---

## Rating Distribution

| Stars | Count | Percentage |
|-------|-------|------------|
| ⭐ 1.0 | 645 | 6.3% |
| ⭐⭐ 2.0 | 519 | 5.1% |
| ⭐⭐⭐ 3.0 | 993 | 9.7% |
| ⭐⭐⭐⭐ 4.0 | 1,967 | 19.2% |
| ⭐⭐⭐⭐⭐ 5.0 | 6,121 | 59.7% |

**Note:** Distribution is skewed toward positive ratings (typical for Amazon reviews)

---

## Usage Examples

### For Person 2 (Baselines)
```python
from src.data_loader import load_splits
import numpy as np

train, test, user_map, item_map = load_splits()

# Global mean baseline
global_mean = train['rating'].mean()
predictions = np.full(len(test), global_mean)

# Evaluate
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test['rating'], predictions)
print(f"Global Mean MSE: {mse:.4f}")
```

### For Person 3 (Matrix Factorization)
```python
from src.data_loader import load_splits
import numpy as np

train, test, user_to_idx, item_to_idx = load_splits()

n_users = len(user_to_idx)
n_items = len(item_to_idx)
n_factors = 10

# Initialize embeddings
user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
item_factors = np.random.normal(0, 0.1, (n_items, n_factors))

# Map IDs to indices
train['user_idx'] = train['user_id'].map(user_to_idx)
train['item_idx'] = train['item_id'].map(item_to_idx)

# Now train matrix factorization...
```

### For Person 4 (Features)
```python
from src.data_loader import load_splits
from sklearn.feature_extraction.text import TfidfVectorizer

train, test, user_map, item_map = load_splits()

# Extract TF-IDF features from review text
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
text_features = vectorizer.fit_transform(train['text'].fillna(''))

print(f"TF-IDF feature matrix shape: {text_features.shape}")
```

---

## Important Notes

 **DO NOT commit raw data files to Git!** They are in `.gitignore`  
 **DO commit** small files like `user_to_idx.pkl` and `item_to_idx.pkl`  
     Everyone uses the same train/test split (seed=42) for fair comparison  
     Mappings ensure consistent indexing across all models  

---

## Data Quality

**Sparsity:** 99.90% - This is expected and realistic for recommender systems!
- Most users review only 2-3 products (median = 2)
- Most products have only 2-3 reviews (median = 2)
- A few power users have many reviews (max = 135)
- A few popular items have many reviews (max = 45)

**Cold Start:**
- All users in test set also appear in train set (no new users)
- All items in test set also appear in train set (no new items)
- This makes evaluation fair and focuses on rating prediction, not cold-start problems

---

## References

- **Official Website:** https://amazon-reviews-2023.github.io/
- **Hugging Face Dataset:** https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **Paper:** Hou, Y., et al. (2024). "Bridging Language and Items for Retrieval and Recommendation"

---
