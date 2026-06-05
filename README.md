# EmoPhone (D1–D3)

> A three-wave in-the-wild smartphone and wearable sensing dataset for moment-level affect modeling in everyday life.

[![Code License](https://img.shields.io/badge/Code%20license-TBD-lightgrey)](./LICENSE)
[![Data Terms](https://img.shields.io/badge/Data%20terms-TBD-lightgrey)](./LICENSE-DATA.md)
[![DOI](https://img.shields.io/badge/DOI-TBD%20on%20acceptance-lightgrey)](#citation)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026%20D%26B-green)](#citation)
[![Datasheet](https://img.shields.io/badge/Docs-Datasheet-orange)](./DATASHEET.md)
[![Croissant](https://img.shields.io/badge/Metadata-Croissant-7f52b5)](./metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json)
[![Harvard Dataverse](https://img.shields.io/badge/Data-Harvard%20Dataverse-red)](#data-access)

> **Dataset name, DOI, paper title, authors, and licenses are finalized on acceptance/public release.** This repository uses the working name `EmoPhone`; final names and legal terms will be substituted throughout before release.

---

## Overview

This repository contains the documentation, metadata, preprocessing notes, and benchmark scaffolding for the **D1, D2, and D3** dataset — three consecutive annual waves of an in-the-wild affective sensing study. Each wave pairs passive sensor data from participants' own Android smartphones and Fitbit wearables with dense in-situ affective state labels collected via the Experience Sampling Method (ESM).

Together, the three waves provide **53,139 ESM responses** from **297 participants** (328 recruited; 9.8% / 11.6% / 7.0% excluded by QC in D1 / D2 / D3), collected between **2020-02-07 and 2022-01-11** in South Korea.

**The dataset is designed to support:**
- Moment-level affect prediction (valence, arousal, stress, task disturbance)
- Within-person (personalized) vs. cross-person generalization benchmarks
- Cross-dataset (cross-wave) transfer-learning experiments
- Longitudinal and individual-difference analyses in mobile affective computing

<p align="center">
  <img src="images/study_design_protocol.png" alt="Three-wave sensing and ESM protocol overview" width="820">
</p>
<p align="center"><em>Figure 1. Three-wave study protocol, sensing streams, and ESM label evolution.</em></p>

---

## Dataset at a Glance

| Property | D1 | D2 | D3 |
|---|---|---|---|
| **Collection period** | Feb 7 - Apr 2, 2020 (~30 d) | Dec 7, 2020 - Jan 27, 2021 (~30 d) | Nov 23, 2021 - Jan 11, 2022 (~28 d) |
| **Recruited / retained (QC)** | 102 / 92 | 112 / 99 | 114 / 106 |
| **ESM responses (post-QC)** | 10,259 | 21,042 | 21,838 |
| **Mean responses / participant** | 111.5 (SD 51.1) | 212.5 (SD 44.1) | 206.0 (SD 24.7) |
| **Smartphone** | Android >= 7.0 | Android >= 8.0 | Android >= 8.0 |
| **Wearable** | Fitbit Inspire HR + Polar H10 (sub-period) | Fitbit Inspire HR + Polar H10 (sub-period) | Fitbit Inspire HR only |
| **Feature columns (total)** | 8,037 | 10,122 | 10,581 |
| **Shared labels** | Valence, Arousal, Stress, Disturbance | Valence, Arousal, Stress, Disturbance | Valence, Arousal, Stress, Disturbance |
| **Wave-specific labels** | Attention, Mental, Duration, Change | Attention, Mental, Duration, ValenceChange, ArousalChange | 8 PANAS-style words (Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry) |

**Shared core labels** (7-point scale, −3 to +3; Stress/Disturbance are 0 to +6 in D3 and normalised to −3/+3 in the benchmark):
Valence, Arousal, Stress, Task Disturbance.

**D3 affect-word labels** (0 to +6): Happy, Relaxed, Cheerful, Content, Sad, Anxious, Depressed, Angry.

See [`data/schema.md`](./data/schema.md) for the full label reference and the column-naming convention.

---

## Benchmark Ladder

| Setting | Scenario | Evaluation | Method families |
|---|---|---|---|
| **Setting A** | Personal-history predictability | 60/20/20 chronological split per user (first 30 days), concatenated across users | Baseline + tabular-NN |
| **Setting B** | Within-dataset cross-user transfer | Stratified group 5-fold by `Pcode`, evaluated per wave | Baseline + tabular-NN + DG + DA |
| **Setting C** | Cross-dataset transfer across waves | Leave-one-dataset-out (1→1 and 2→1), shared labels only, common-feature intersection | Baseline + tabular-NN + DG + DA |

All settings use **AUROC** as the primary metric with Accuracy / Macro-F1 / Precision / Recall reported for diagnostics. Hyperparameters are tuned with **Optuna (30 trials, validation-AUROC selection)**; training uses a unified loop (≤ 50 epochs, patience-based early stopping, fixed seed).

See [`benchmark/README.md`](./benchmark/README.md) for setting-level details and reproduction pointers.

### Benchmark Model Inventory

**Baselines**: XGBoost, LightGBM, MLP, ResNet.
**Tabular neural networks**: TabNet, SAINT, TabTransformer, FTTransformer, DCN.
**Domain generalization (DG)**: IRM, VREx, GroupDRO, MixStyle, MLDG, MASF, Fish, CSD, SagNet.
**Domain adaptation (DA)**: DANN, CDAN, DAN, DeepCORAL, MCC, ADDA, MCD, JAN, SHOT, CBST, CGDM.

DG and DA models share an MLP backbone so that family differences reflect the objective, not the backbone. DG methods follow DomainBed protocols; DA methods follow the Transfer-Learning-Library (TLL); tabular NNs follow their respective upstream repositories.

## Figure Assets

The README keeps the study-protocol figure inline and groups the remaining reviewer-facing visuals here so the main flow stays readable. All image assets live under [`images/`](./images/) with stable, lowercase filenames.

<details>
<summary>Open reviewer figure gallery</summary>

| Dataset characteristics | Sensor and label coverage |
|---|---|
| <img src="images/dataset_characteristics.png" alt="Dataset characteristics table" width="390"> | <img src="images/sensor_label_matrix.png" alt="Sensor and label coverage matrix" width="390"> |

| ESM response timing | Cross-user embedding structure |
|---|---|
| <img src="images/temporal_distribution_esm_labels.png" alt="Temporal distribution of ESM labels" width="390"> | <img src="images/tsne_cross_user.png" alt="Cross-user embedding structure" width="390"> |

| Cross-wave embedding structure |
|---|
| <img src="images/tsne_cross_wave.png" alt="Cross-wave embedding structure" width="390"> |

</details>

---

## Preprocessing Pipeline

All three waves were processed using the reproducible mobile-sensing pipeline introduced in:

> Zhang, P., Jung, G., Alikhanov, J., Ahmed, U., & Lee, U. (2024). **A Reproducible Stress Prediction Pipeline with Mobile Sensor Data.** *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.* 8(3). https://doi.org/10.1145/3678578

Per-wave QC thresholds, feature-alignment decisions, and scale-normalisation rules are documented in [`preprocessing/pipeline_decisions.md`](./preprocessing/pipeline_decisions.md) and [`docs/feature_alignment.md`](./docs/feature_alignment.md).

---

## Repository Layout

```
EmoPhone/
│
├── README.md                         ← this file
├── DATASHEET.md                      ← Gebru-style datasheet (NeurIPS D&B requirement)
├── LICENSE                           ← code license placeholder (TBD)
├── LICENSE-DATA.md                   ← data license / DUA placeholder (TBD)
├── CITATION.cff                      ← machine-readable citation
├── AUTHORS.md                        ← contributors
├── RESPONSIBILITY.md                 ← NeurIPS D&B author responsibility statement
├── MAINTENANCE.md                    ← hosting, versioning, SLA
├── CHANGELOG.md                      ← release history
├── CONTRIBUTING.md                   ← PR / issue guidance
├── requirements.txt                  ← top-level pinned Python deps
├── environment.yml                   ← optional conda mirror
├── pyproject.toml                    ← Black / isort / ruff config
│
├── docs/                             ← long-form docs
│   ├── dataset_overview.md
│   ├── feature_alignment.md          ← cross-wave feature schema & alias map
│   ├── ethics.md
│   ├── consent_form_en.md            ← translated consent form (placeholder)
│   └── neurips_db_checklist.md       ← reviewer-facing compliance checklist
│
├── data/
│   ├── README.md                     ← Dataverse access + how to load files
│   └── schema.md                     ← column-by-column reference
│
├── preprocessing/
│   ├── README.md                     ← pipeline overview & Zhang 2024 reference
│   ├── pipeline_decisions.md         ← per-wave QC, scale normalisation, alias map
│   ├── feature_alignment.md          ← mirrors docs/feature_alignment.md
│   ├── 01_preprocessing/             ← per-wave app-categorisation + ESM cleanup notebooks
│   ├── 02_feature_extraction/        ← per-wave feature + label-sync extraction notebooks
│   └── common/                       ← shared sliding-window utilities used across waves
│
├── benchmark/                        ← protocol specs + result tables (docs only)
│   ├── README.md                     ← three-setting ladder + model inventory
│   ├── setting_a/README.md           ← personal-history predictability
│   ├── setting_b/README.md           ← within-wave cross-user transfer
│   ├── setting_c/README.md           ← cross-wave transfer
│   ├── utils/README.md               ← shared loader / metric contract
│   └── results/                      ← committed per-task CSV/JSON summaries
│
├── basemodel-benchmarking/           ← Setting A/B/C baseline + tabular-NN runtime
│   ├── README.md
│   ├── benchmark.py                  ← Setting A + Setting B runner
│   ├── cross_dataset.py              ← Setting C runner
│   ├── benchmark_logger.py
│   ├── data_loader.py, hparams_registry.py, backbones.py, utility.py
│   ├── models/                       ← per-wrapper baseline + tabular-NN package
│   │   ├── baselines.py, xgb.py, lgb.py, tabnet.py, widedeep.py, deepctr.py
│   │   └── _helpers.py
│   └── scripts/                      ← shell drivers (run_setting_{a,b,c}.sh, run_debug.sh)
│
├── domain_adaptation/                ← Setting B/C DG + DA runtime
│   ├── README.md
│   └── models/
│       ├── _da_helpers.py
│       ├── da_tllib_losses.py
│       ├── da_models.py              ← back-compat shim → da/*
│       ├── domainbed_algos.py        ← back-compat shim → dg/*
│       ├── da/                       ← one file per DA algorithm
│       └── dg/                       ← one file per DG algorithm
│
├── EDA/                              ← dataset-characterisation notebooks
├── images/                           ← figures referenced by READMEs
└── metadata/
    ├── croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json
    │                                ← ML Commons Croissant metadata
    └── checksums.md5                 ← per-archive SHA/MD5 (populated on release)
```

---

## Data Access

The planned release target is **Harvard Dataverse** with gated access. Users will log in, agree to the Data Use Agreement (DUA), and then download. The final landing page, file IDs, DOI, license, and DUA text will be populated on acceptance/public release.

**To download:**
1. Visit the dataset page at **[TBD Harvard Dataverse URL — populated on acceptance]**.
2. Log in or create a free Harvard Dataverse account.
3. Read and agree to the Data Use Agreement.
4. Download individual wave archives (`D1.zip`, `D2.zip`, `D3.zip`) or the full dataset.

**Download via the Dataverse API** (after agreeing to terms on the website):

```bash
# D1 as an example
curl -L "https://dataverse.harvard.edu/api/access/datafile/TBD_FILE_ID_D1" \
     -H "X-Dataverse-key: YOUR_API_TOKEN" \
     -o D1.zip
```

**Python equivalent**:

```python
import requests

api_token = "YOUR_API_TOKEN"  # Dataverse account → API Token
file_id   = "TBD_FILE_ID_D1"  # from the Dataverse dataset page

r = requests.get(
    f"https://dataverse.harvard.edu/api/access/datafile/{file_id}",
    headers={"X-Dataverse-key": api_token},
)
with open("D1.zip", "wb") as f:
    f.write(r.content)
```

See [`data/README.md`](./data/README.md) for the full data-folder structure and loading examples.

**Verify integrity after download:**
```bash
md5sum -c metadata/checksums.md5
```

---

## Reproducibility

- **Python**: 3.10 recommended; 3.9–3.11 supported.
- **Install**: `pip install -r requirements.txt` (or `conda env create -f environment.yml`).
- **Seeds**: all benchmark runs use a fixed seed (documented per setting in [`benchmark/setting_*/README.md`](./benchmark/)).
- **HPO**: Optuna with **30 trials**, validation-AUROC selection, setting-specific validation split.
- **Training loop**: maximum 50 epochs, patience-based early stopping, unified across baseline / DG / DA families.
- **Split definitions**, preprocessing policy, and model-selection rule are held fixed across families within each setting.

---

## Citation

```bibtex
@inproceedings{emophone2026,
  title     = {[Dataset Title — TBD on acceptance]},
  author    = {[Author list — TBD on acceptance]},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2026},
  doi       = {TBD},
  url       = {TBD}
}

@article{zhang2024reproducible,
  title     = {A Reproducible Stress Prediction Pipeline with Mobile Sensor Data},
  author    = {Zhang, Panyu and Jung, Gyuwon and Alikhanov, Jumabek and Ahmed, Uzair and Lee, Uichin},
  journal   = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  volume    = {8},
  number    = {3},
  year      = {2024},
  doi       = {10.1145/3678578}
}
```

See [`CITATION.cff`](./CITATION.cff) for the machine-readable form.

---

## License and Ethics

- **Code license:** TBD; see [`LICENSE`](./LICENSE).
- **Data license and DUA:** TBD; see [`LICENSE-DATA.md`](./LICENSE-DATA.md).

All data collection was approved by the Institutional Review Board at **[TBD institution]** (approval number **[TBD]**). Participants provided written informed consent and received ~100 USD compensation. Anonymisation applied prior to release: MD5-hashed contact numbers, UUID-replaced Wi-Fi/Bluetooth MAC addresses, per-participant random displacement of GPS longitude. See [`docs/ethics.md`](./docs/ethics.md) and [`DATASHEET.md`](./DATASHEET.md) § 2.

The authors accept full responsibility for any rights violations arising from this release; see [`RESPONSIBILITY.md`](./RESPONSIBILITY.md).

Maintenance plan and versioning policy: [`MAINTENANCE.md`](./MAINTENANCE.md). Change history: [`CHANGELOG.md`](./CHANGELOG.md).

---

## Contact

For questions about the dataset or code, contact **[TBD contact email — populated on acceptance]** or open a GitHub issue. Contribution guidance: [`CONTRIBUTING.md`](./CONTRIBUTING.md).
