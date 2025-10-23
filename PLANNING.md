# UbiComp Experiment Planning

This plan expands the **Experiment Design → Pretraining** notes from `UbiComp_Doc_Personalization.docx` into actionable work items. The goal is to iteratively implement and validate cross-dataset stress detection experiments inside the `Ubicomp` project.

## 1. Objectives
- Predict *valence*, *arousal*, *stress*, and *attention* labels under pronounced cross-user/domain shifts.
- Leverage existing CrossShift-style models (XGBoost, LightGBM) and extend to tabular generalization baselines (SAINT, TabTransformer, NODE, FT-Transformer, TransBoost, etc.).
- Evaluate transferability with **LOSO** (leave-one-subject-out) and **stratified group 5-fold** protocols.
- Compare *pretraining + fine-tuning* vs. *target-only* training, and quantify the impact of dataset combinations (D-1, D-2, D-3).

## 2. Datasets & Preprocessing
| Alias | Source pickle (default) |
| ----- | ----------------------- |
| D-1 | `~/minseo/Archived/stress_binary_personal-current_D#2.pkl` |
| D-2 | `~/minseo/Archived/stress_binary_personal-current_D#3.pkl` |
| D-3 | `~/minseo/Archived/stress_binary_personal-current.pkl` |

**Tasks**
1. [ ] Confirm paths and availability; override via CLI if needed (`--dataset-override`).
2. [x] 표준화: 모든 데이터셋을 교집합 피처만 사용하도록 정렬 (`domain_adaptation.data_utils.align_feature_intersection`).
3. [ ] Track metadata (timestamp ranges, user counts, label distribution) for reproducibility.

## 3. Pretraining + Fine-tuning Experiments
### 3.1 Scenario Matrix
| Pretrain Sources | Target |
| ---------------- | ------ |
| D-1 + D-3 | D-2 |
| D-1 + D-2 | D-3 |
| D-2 + D-3 | D-1 |

For each scenario:
- Pretrain on concatenated source datasets.
- Fine-tune on target training split; evaluate on target validation/test.
- Compare against *target-only* training with identical hyperparameters.

### 3.2 Implementation Steps
1. [ ] Parameterize `domain_adaptation.pipeline.run_experiment_scenario` to enforce LOSO and stratified group 5-fold splits. **추가**: D-1, D-2, D-3 각각을 독립 실행하는 별도 진입점/스크립트를 마련해 `D-1+D-3→D-2`와 같은 복합 시나리오 외에도 dataset 단위 결과를 남긴다.
2. [x] Extend pipeline outputs to include per-class metrics (AUROC/PRAUC/Accuracy) for all four targets.
3. [ ] Add configuration for XGBoost pipeline mirroring CrossShift settings (learning rate, max depth, estimators).
4. [x] Log pretrain/fine-tune sample counts, training times, and best iteration numbers for later analysis.
5. [ ] Persist experiment summaries to `results/domain_adaptation/` with scenario + seed metadata.


`scripts/experiments/pretrain_transfer.py`에 `--split-strategy`, `--group-folds`, `--val-size`, `--test-size` 옵션을 추가하여 CLI에서도 동일한 분할 구성을 제어할 수 있다.

## 4. Evaluation Protocols
- **LOSO**: treat each user as held-out fold; requires user-level grouping (integrate with `GroupKFold` or custom splitter).
- **Stratified Group 5-Fold**: enforce both label stratification and user grouping (can adapt `StratifiedGroupKFold` from `sklearn.model_selection` in 1.3+ or implement manual splitter).
- Metrics: AUROC, AUPRC, accuracy, confusion matrix per fold, plus aggregated mean ± std.
- Record OTDD distances (via `utility.calculate_user_similarity_ranking`) to correlate transfer performance with domain shifts.

**Tasks**
1. [x] Implement reusable splitter utilities under `domain_adaptation/data_utils.py` to serve both tree and transformer pipelines.
2. [x] Update experiment logs to attach splitter type and random seed.

## 5. Domain Adaptation Baselines
### 5.1 TransBoost (as cited)
1. [ ] Add TransBoost implementation or vendor package integration under `domain_adaptation/models/transboost.py`.
2. [ ] Define training interface aligned with existing pipeline (pretrain/adapt/eval hooks).
3. [ ] Benchmark against LightGBM & transformer baselines on the scenario matrix.

### 5.2 Dataset Combination Ablations
- Compare for each fine-tune 비율(0%, 20%, …):
  - Combined (D-1 + D-3) pretraining + D-2 fine-tune/test.
  - Target-only D-2 training (pretrain 없이 fine-tune 데이터만으로 학습).
  - Combined datasets without pretraining (direct pooled training).
- For each, report metrics and delta vs. target-only baseline.

## 6. Tabular Generalization Models
**Models to include (per doc + packages):**
- SAINT, TabTransformer, NODE, FT-Transformer, Tabular ResNet (from TabBenchmark).
- Additional Domain Robustness / Label Shift / Domain Generalization models from GLOBEM packages.

**Tasks**
1. [ ] Audit external packages (GLOBEM, TabBenchmark, Domain Robustness, Label Shift) and catalog model + dependency requirements.
2. [ ] Create wrappers under `domain_adaptation/models/` mirroring LightGBM/Transformer interfaces.
3. [ ] Standardize config schemas (YAML/JSON or dataclasses) for hyperparameters.
4. [ ] Design experiment runner capable of batching models for each scenario and saving per-model CSV outputs.

## 7. Analysis & Reporting
- Produce `results/domain_adaptation/dashboard.csv` tracking:
  - Scenario, model, mode (pretrain_finetune vs. target_only).
  - Metric aggregates + OTDD correlation coefficients.
  - Training time and resource footprint (threads, GPU usage if applicable).

## 8. Implementation Phases
1. **Phase 0 – Infrastructure**
   - [ ] LOSO + StratifiedGroup splitters (dataset별 독립 실험 및 pretrain/target 분할 로깅 포함).
   - [x] Enhanced logging & results schema.
2. **Phase 1 – XGBoost Pretraining**
   - [x] Integrate XGBoost pipeline (random split end-to-end run logged in `results/pipeline_run_log.csv`).
     - Latest baseline: 2025-10-12 random 80/20 split (`model=xgb`, 46 anchors, 40 features) – see `distance_figures_xgb/` outputs for plots and CSV summaries.
   - [ ] Run scenario matrix, validate metrics (LOSO/StratifiedGroupKFold pending – current pipeline assumes per-user splits).
3. **Phase 2 – Domain Adaptation Extensions**
   - [ ] Add TransBoost.
   - [ ] Implement dataset combination experiments.
4. **Phase 3 – Tabular Generalization Sweep**
   - [ ] Wire in TabBenchmark models.
   - [ ] Automate comparison runs.
5. **Phase 4 – Analysis & Reporting**
   - [ ] Aggregate results, compute OTDD correlations.
   - [ ] Generate summary tables/plots for manuscript or model handoff.

## 9. Open Questions / Dependencies
- Confirm whether label sets (valence/arousal/stress/attention) are mutually exclusive or multi-label; adjust training targets accordingly.
- Determine GPU availability for transformer/tabular models and TransBoost.
- Validate that OTDD computation scales with combined datasets (adjust subsampling thresholds if required).
- Cross-check overlap between GLOBEM and TabBenchmark model lists to avoid duplicate effort.
- Optional dependency log (updated 2025-10-12): installed `statsmodels`, `ray`, `tqdm_joblib`, `xgboost`, `lightgbm`, `catboost`, `otdd` (microsoft/otdd). OTDD 모듈은 `from otdd.pytorch.distance import DatasetDistance` 임포트 경로를 사용하며, 이 경로 변경 시 전 파이프라인 업데이트 필요.
- **Policy reminder:** keep “fail fast” discipline across codepaths (no fallback behaviour; surface exceptions and iterate until fixed) so future models inherit the same guarantees.

---
**Next Step:** Extend the pipeline to handle LOSO/StratifiedGroupKFold splits (shared-normalization fix) and populate the XGBoost scenario matrix.
