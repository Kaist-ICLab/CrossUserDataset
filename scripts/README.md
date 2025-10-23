# Analysis Scripts

Organized Python scripts for stress detection analysis.

## Structure

### `preprocessing/`
- `create_dataset.py` - Extract selected users from original dataset
- `reduce_features.py` - Create reduced 49-feature dataset

### `training/`
- `train_models.py` - Train LightGBM models with feature importance

### `analysis/`
- `compare_performance.py` - Compare full vs reduced feature performance
- `feature_analysis.py` - Analyze feature importance patterns

## Usage

**Environment:** Always use `conda activate navsim`

**Run complete pipeline:**
```bash
# 1. Create datasets
python scripts/preprocessing/create_dataset.py
python scripts/training/train_models.py
python scripts/preprocessing/reduce_features.py

# 2. Compare performance
python scripts/analysis/compare_performance.py

# 3. Analyze features
python scripts/analysis/feature_analysis.py
```

## Dependencies
- LightGBM, Optuna, scikit-learn, pandas, numpy
- All scripts follow exact `stress_ml_pipeline.py` methodology