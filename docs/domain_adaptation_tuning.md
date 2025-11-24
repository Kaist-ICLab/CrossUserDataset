## Domain Adaptation Transformer – Issues & Fixes

### 1. Feature Explosion & Random Weights
**Per-User Normalization**
- Sensors are z-scored within each PID (`value - user_mean` divided by user std, fallback to 0 when std=0). This happens inside `augment_temporal_and_group_features` before global StandardScaler, so every sensor now reflects “deviation from that user's baseline” rather than absolute magnitude.  

### 2. Dataset Size & Scaling Instability
- **Issue:** StandardScaler trained only on sources could not handle large shifts, yielding infinities.  
- **Fixes:**  
  - Option to include target in scaler (`--include-target-in-scaler`).  
  - Post-scaling clipping (`--feature-clip-value`).  
  - Stronger feature filtering knobs above.

### 3. Pandas Fragmentation Warnings
- **Issue:** Repeated `df[col] = ...` inserts inside `augment_temporal_and_group_features` triggered “DataFrame is highly fragmented” warnings and slowed preprocessing.  
- **Fix:** Build derived columns (`PID_FEATURE`, TOD/DOW sin/cos, rolling stats, user meta features) via one `pd.concat`.

### 4. User-Level Drift
- **Issue:** Different users exhibit distinct stress baselines, hurting transfer.  
- **Fix:** Added user meta features (sample counts, log counts, activity levels, prefix-wise means) plus rolling 24h/72h stats and deltas to give the model richer per-user context.

### 5. Training Plateau
- **Issue:** Fixed LR/epoch schedule plateaued near AUROC 0.70.  
- **Fix:** Added scheduler knobs (`use_cosine_scheduler`, `warmup_epochs`, `use_plateau_scheduler`) and exposed hyperparameters via `--transformer-option`.

### 6. Scenario Control
- **Issue:** Running all dataset combinations was slow while debugging a single transfer path.  
- **Fix:** `--scenarios 'SRC1+SRC2->TGT'` argument in the CLI to run only the combinations you care about.

### 7. Workflow Tips
- Use caching on SSD (`--cache-dir`) to amortize preprocessing.  
- Start with aggressive filtering (`min_source_feature_frac≈0.05`, `max_target_shift_std≈10`, `max_feature_correlation≈0.98`) then relax as needed.  
- Validate source pretrain AUROC first (since fine-tune ratios are 0.0).  
- Keep an eye on `results/domain_adaptation_results.csv` for `feature_count`, `pretrain_val_auroc`, and runtime columns to compare experiments.
