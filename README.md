# Stress Detection: High-Performance Users Analysis

## Overview
Analysis of 13 high-performance users selected from validation/test metrics ≥80% in at least one metric.

## Selected Users
P124, P135, P045, P050, P052, P071, P133, P046, P056, P048, P024, P094, P105

## Directory Structure

### `/selected_users_dataset/`

**Main Datasets:**
- `full_216features.pkl` - Full dataset (13 users, 216 features)
- `reduced_49features.pkl` - Reduced dataset (13 users, 49 features)
- `feature_mapping.pkl` - Feature mapping info

**Results Analysis:**
- `results/` - Feature importance analysis results
  - `feature_summary.csv` - All users feature importance summary
  - `performance_features.csv` - Performance + top features per user
  - `P{ID}_feature_importance.csv` - Individual user feature importance

## Key Findings

### Feature Reduction
- **77% reduction**: 216 → 49 features
- **98% performance retention** on average
- Top users maintain >100% performance with reduced features

### Performance (Test AUROC)
- **Mean**: 0.818 ± 0.075 (full), 0.799 ± 0.080 (reduced)
- **Best users**: P135 (0.927), P124 (0.922), P045 (0.917)
- **Feature efficiency**: 49 features capture 89% of importance

### Most Important Features (All Users)
1. feature_63, feature_212, feature_64 (top 3)
2. 17 features important for all 13 users
3. 49 features cover 99% of user-specific needs

## Usage

**Environment:** Always use `conda activate navsim`

**Load datasets:**
```python
# Full dataset
with open('selected_users_dataset/full_216features.pkl', 'rb') as f:
    X, y, users, timestamps, feature_names = pickle.load(f)

# Reduced dataset
with open('selected_users_dataset/reduced_49features.pkl', 'rb') as f:
    X_reduced, y, users, timestamps, feature_names = pickle.load(f)
```

## Workflow

Follow these steps whenever you refresh datasets, models, feature importances, or cross-user analysis assets:

### 0. Rebuild the selected-users dataset (optional)

If you want to choose the user subset dynamically—e.g., drop users whose class
imbalance exceeds 3× or restrict to an explicit user list—run:

```bash
python scripts/preprocessing/create_dataset.py \
  --source-pkl ~/Archived/stress_binary_personal-current.pkl \
  --selected-users-file selected_ids.txt \
  --max-label-ratio 3.0
```
*(omit `--selected-users-file` to keep every user in the source dataset)*

`create_dataset.py` loads the raw dataset, optionally filters by user ID and
label imbalance, and rewrites both `selected_users_dataset/full_216features.pkl`
and `full_216features_normalized.pkl` with the surviving users.

### 1. Train per-user models on the full feature set

```bash
python scripts/training/train_models.py \
  --dataset-path selected_users_dataset/full_216features_normalized.pkl \
  --raw-dataset-path selected_users_dataset/full_216features.pkl \
  --selection-threshold 0.8 \
  --save-models \
  --user-jobs 4
```

- Outputs
  - `selected_users_dataset/results/*.csv` (per-user & summary importances)
  - `selected_users_dataset/models/full_216features_normalized/<USER>/` (fold LightGBM models, written in parallel)
  - Datasets rewritten to include only users achieving `val_auroc ≥ selection-threshold`
  - Use `--threads-per-model` if you prefer to pin LightGBM thread count manually (default derives from CPU/user-jobs)

### 2. Reduce features and retrain on the 49-feature subset

```bash
python scripts/preprocessing/reduce_features.py
```

- Builds `reduced_49features.pkl` and reuses the pre-normalized full dataset to emit `reduced_49features_normalized.pkl`
- Retrains per-user LightGBM models on the reduced data (saved under `selected_users_dataset/models/reduced_49features_normalized/<USER>/`)
- Stores updated CSVs in `selected_users_dataset/results/reduced/`

### (Optional) 2b. Generate leaf-index embeddings for OTDD

```bash
python scripts/preprocessing/generate_leaf_embeddings.py \
  --dataset-path selected_users_dataset/reduced_49features_normalized.pkl \
  --output-path selected_users_dataset/reduced_49features_leaf.pkl \
  --num-boost-round 500 \
  --num-leaves 64 \
  --num-threads 8
```

- Trains (or loads) a global LightGBM model and converts each sample into a one-hot leaf embedding.
- The resulting `*_leaf.pkl` files can be loaded via `load_selected_dataset('reduced_leaf')` when computing OTDD.

### 3. OTDD distances & utilities

`utility.py` exposes helpers that consume the saved models/importances:

```python
from utility import load_selected_dataset, calculate_user_similarity_ranking

X, y, users, _, feature_names = load_selected_dataset('reduced')
features_df = pd.DataFrame(X, columns=feature_names)

ranking = calculate_user_similarity_ranking(
    train_user='P124',
    features=features_df,
    labels=y,
    group_indices=users,
    feature_list=feature_names,
    results_dir='selected_users_dataset/results',
    results_subdir='reduced',
    cache_path='otdd_distances_P124.npy'
)
```

### 4. Cross-user evaluation using saved models

```bash
python cross_user_evaluation.py P124 Ubicomp/otdd_distances_P124.npy reduced_49features_normalized
```

- `train_user` (arg 1): source user to transfer from
- `distance_matrix_path` (arg 2): OTDD matrix produced via `utility.py`
- `model_dataset_tag` (arg 3): dataset tag matching the saved model bundle (e.g., `reduced_49features_normalized`)

Cross-user evaluation loads the persisted boosters + scaler, reports metrics against all other users, and writes `P124_cross_user_evaluation.csv`.

## 실행 요약 (한글)

0. **선택 사용자 재구성(선택)** – `create_dataset.py`를 실행해 원본 데이터를 정규화 버전까지 생성하고, 필요하다면 사용자 ID나 라벨 비율로만 필터링한다.
1. **풀 피처 학습** – `train_models.py --selection-threshold <값> --user-jobs <병렬수>`를 실행하면 Optuna 학습 → AUROC 기준 사용자 선별 → 필터링된 데이터셋 재생성이 한 번에 진행되고, 최종 모델/결과가 `results/`와 `models/full_216features_normalized/`에 저장된다 (`--threads-per-model`로 LightGBM 스레드 수 조절 가능).
2. **피처 축소/재학습** – `python scripts/preprocessing/reduce_features.py` 실행 → 49개 피처 데이터(정규화 포함)를 생성하고 재학습 결과가 `results/reduced/`, `models/reduced_49features_normalized/`에 저장.
   (선택) `scripts/preprocessing/generate_leaf_embeddings.py`로 Leaf 임베딩(`*_leaf.pkl`)을 생성하면 OTDD 계산 시 모델 기반 표현을 사용할 수 있음.
3. **OTDD 계산** – `utility.py`의 `calculate_user_similarity_ranking` 등을 사용하여 OTDD 거리 행렬(`*.npy`) 생성. 이때 필요한 사용자별 중요도/모델은 위에서 저장한 리소스를 사용.
4. **교차 사용자 평가** – `python cross_user_evaluation.py <훈련유저> <OTDD행렬경로> reduced_49features_normalized` 형태로 실행하여 저장된 모델을 불러와 transfer 성능을 평가.

## OTDD Distance Calculation

**IMPORTANT:** The OTDD distance calculation in `utility.py` has **NO FALLBACK** mechanisms. If OTDD computation fails for any reason (memory issues, PyTorch version incompatibility, etc.), the function will **RAISE AN EXCEPTION** and terminate. This is by design.

- **No Wasserstein fallback**
- **No default distance values**
- **Fail fast approach**: If OTDD doesn't work, fix the underlying issue rather than masking it
- OTDD 라이브러리는 `navsim` 환경에 설치된 [microsoft/otdd](https://github.com/microsoft/otdd/tree/main/otdd) 레포지토리 버전을 사용하며, 코드에서는 `from otdd.pytorch.distance import DatasetDistance` 형태로 임포트한다.

**Supported configurations:**
- 49-feature datasets with 49-feature importance weights
- 216-feature datasets with 216-feature importance weights
- **NO cross-dimensional mapping** (e.g., no 216→49 feature mapping)

## Domain Adaptation Pretraining Pipeline

The new `domain_adaptation` package bundles utilities to pretrain on two
datasets, fine-tune on a target dataset, and sweep fine-tuning ratios. Key
pieces:

- `domain_adaptation/data_utils.py` — dataset loading and feature intersection alignment
  (cached under `.cache/`).
- `domain_adaptation/models/` — LightGBM and transformer training pipelines
  with staged pretrain → fine-tune → adapt routines.
- `domain_adaptation/pipeline.py` — experiment orchestration and result
  aggregation.
- `scripts/experiments/pretrain_transfer.py` — CLI entrypoint.

### Requirements

Activate the `navsim` conda 환경 before running any scripts, and ensure the
following packages are available in that environment:

- `lightgbm`
- `torch` (CUDA optional)

### Default scenarios

The CLI evaluates the requested combinations, mapping D-1/2/3 to
`~/minseo/Archived/` pickles by default:

- `D-1 + D-3 → D-2`
- `D-1 + D-2 → D-3`
- `D-2 + D-3 → D-1`

모든 데이터셋은 공통 피처 교집합으로 정렬되며, 그 공간에서 학습 및 평가가
진행된다. 결과는 지정된 미세조정 비율(0–60%)과 네 개의 시드에 대해 LightGBM과
트랜스포머 모델을 모두 기록한다.
- 결과 CSV에는 AUROC뿐 아니라 Accuracy, AUPRC, 학습 단계별 소요 시간 및 LightGBM best iteration 메타데이터가 포함된다.

### Running the sweep

```bash
python scripts/experiments/pretrain_transfer.py \
  --cache-dir .cache/domain_adaptation \
  --output-csv results/domain_adaptation_results.csv \
  --fine-tune-ratios 0.0 0.2 0.4 0.6 \
  --fine-tune-val-ratio 0.2 \
  --seeds 42 43 44 45 \
  --model-types tree transformer \
  --pretrain-val-ratio 0.1 \
  --split-strategy stratified_group_kfold \
  --group-folds 5
```

Override dataset locations if needed with `--dataset-override`
(`D-1=/custom/path.pkl`). Every run records both the pretrain+fine-tune workflow
and the target-only baseline for comparison.

- `--split-strategy`는 `stratified_shuffle`, `loso`, `stratified_group_kfold` 중 하나를 선택해 타깃 분할 방식을 제어한다.
- `--group-folds`는 `stratified_group_kfold` 사용 시 폴드 개수를 지정하며, `--val-size`, `--test-size`는 `stratified_shuffle` 전략에서만 사용된다.
- `--pretrain-val-ratio`로 소스(Pretrain) 데이터 검증 비율을 직접 지정할 수 있고(기본 10%), `--fine-tune-val-ratio`는 타깃 적응 데이터 내부에서 validation으로 떼어둘 비율을 제어한다. 필요하면 `--target-train-ratio / --target-val-ratio / --target-test-ratio`로 타깃 전체의 분할 비율을 명시적으로 고정할 수 있다(명시되지 않으면 `--val-size`, `--test-size` 기반 기본 분할 사용).