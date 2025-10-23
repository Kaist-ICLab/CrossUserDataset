# UbiComp Progress Log

## Completed (현재 진행 상황)
- `domain_adaptation/data_utils.py`에 LOSO 및 StratifiedGroupKFold 분할 유틸 추가.
- `domain_adaptation/pipeline.py`가 분할 전략(`stratified_shuffle`, `loso`, `stratified_group_kfold`)을 선택적으로 지원하고, 결과 로그에 `split_strategy`, `fold_id`, `test_groups`, `split_seed` 메타데이터를 포함하도록 리팩터링됨.
- `scripts/experiments/pretrain_transfer.py` CLI에 새 분할 옵션(`--split-strategy`, `--group-folds`, `--val-size`, `--test-size`) 노출.
- `README.md`, `PLANNING.md`에 최신 OTDD 임포트 경로(`from otdd.pytorch.distance import DatasetDistance`) 및 신규 분할 옵션 문서화.
- LightGBM/Transformer 파이프라인 결과에 Accuracy·AUPRC·학습 단계별 소요 시간 및 (LightGBM) best iteration 메타데이터를 추가하고, CSV 출력에도 해당 값들이 포함되도록 확장.
- 샘플 실험 실행 완료: `--split-strategy loso`(LightGBM 전용, `results/domain_adaptation_loso.csv`, 594행)와 `--split-strategy stratified_group_kfold`(5-fold, `results/domain_adaptation_stratified_group.csv`, 30행)로 메타데이터 저장형식 검증.
- D-1에서만 존재하던 한글/동의어 피처명을 영어 기반으로 정규화하고 중복 컬럼은 합산 처리하여 교집합 114개, 잔여 고유 피처 330개로 재산출 (`results/feature_diff.csv` 재생성).
- SHAP 기반 피처 선택을 제거하고 세 데이터셋 공통 교집합 피처만 사용하도록 정렬 로직을 단순화.
- 소스 데이터 pretrain 시 독립 validation split(기본 10%) + pretrain validation 메트릭 로깅을 추가하고, 타깃 데이터는 `--target-*-ratio`와 `--fine-tune-val-ratio`로 train/val/test 및 적응 검증 비율을 제어하도록 파이프라인/CLI 강화. 재실행 결과는 `results/domain_adaptation_default.csv`에 저장(예: train 80%, val 10%, test 10%).

## Decisions & Notes
- OTDD 패키지는 `navsim` Conda 환경에 설치된 microsoft/otdd 버전을 사용해야 하며, “fail fast” 정책에 따라 모든 거리 계산은 예외를 숨기지 않는다.
- 분할 전략은 기본적으로 `stratified_shuffle`이나, LOSO/StratifiedGroupKFold 실험을 위한 인프라가 준비되어 있음.

## Next Actions (다음 단계)
1. `results/domain_adaptation_loso.csv` 및 `results/domain_adaptation_stratified_group.csv`를 기반으로 `cross_user_evaluation.py` 결과와 비교 가능한 대시보드/요약 스키마 초안 작성.
2. LightGBM/Transformer 외 추가 모델을 통합하고 동일 지표/메타데이터를 기록하도록 파이프라인 확장.
3. 장기적으로는 OTDD 결과와 성능 지표를 결합한 분석 노트북을 작성해 도메인 간 전이 성능을 시각화.


## Environment Reminder
- 모든 스크립트는 `conda activate navsim` 활성화 상태에서 실행.
