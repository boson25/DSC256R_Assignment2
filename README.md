
---

## Project Overview

**Task:** Predict user ratings (1-5 stars) for Amazon Beauty product reviews

**Dataset:** Amazon Reviews 2023 - All_Beauty category
- 10,245 reviews (after preprocessing)
- 3,543 unique users
- 2,998 unique items
- 99.90% sparsity

**Models:**
- **Baselines:** Global mean, user mean, item mean, bias model
- **Advanced:** Matrix factorization with SGD, regularization tuning
- **Features:** TF-IDF text features, temporal features, item metadata


## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/DSR256R_Assignment2.git
cd DSC256R_Assignment2
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Load Preprocessed Data
```python
import sys
sys.path.append('src')
from data_loader import load_splits

# Load data
train, test, user_map, item_map = load_splits()

```

---

## Data Access

Preprocessed data is available in `data/processed/` 


## Technical Details

**Programming Language:** Python 3.11+

**Key Libraries:**
- pandas, numpy - Data manipulation
- scikit-learn - ML models and metrics
- matplotlib, seaborn - Visualization
- jupyter - Notebooks

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**Baseline Models:**
- Global mean predictor
- User mean predictor
- Item mean predictor
- User + Item bias model

### Baseline Results
We evaluated four baseline models to establish reference performance before moving to more advanced methods. All evaluations were done on the processed train/test split using MSE.

| Model | Test MSE |
|-------|----------|
| Global Mean | 1.559 |
| User Mean | 1.912 |
| Item Mean | 1.987 |
| **Bias Model (User + Item Bias)** | **1.437** |

**Summary:**
- The **global mean** provides a simple starting point.
- **User mean** and **item mean** baselines reflect individual tendencies, but perform worse.
- The **bias model** captures both user and item deviations from the global average and achieves the lowest error among all baselines.
- This model will serve as the primary reference for evaluating more complex models in later stages.


**Advanced Models:**
- Matrix factorization with SGD
- Regularized latent factor models
- Feature-enhanced models

---

## Resources

- **Dataset:** https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
