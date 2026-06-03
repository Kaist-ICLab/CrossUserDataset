# Domain Generalization & Domain Adaptation

This folder hosts the runnable code for the **domain generalisation (DG)** and **domain adaptation (DA)** baselines used in Setting B and Setting C.

> **Conceptual documentation lives in [`../benchmark/`](../benchmark/).** This folder is the implementation side; the `benchmark/` folder describes the protocol each setting expects.

---

## Scope

Runs in this folder produce results for:

- **Setting B** — within-dataset cross-user transfer, DG and DA families ([`../benchmark/setting_b/README.md`](../benchmark/setting_b/README.md)). "Domains" are user-clusters within a single wave.
- **Setting C** — cross-dataset transfer, DG and DA families ([`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md)). Source / target domains are the waves themselves (1→1 and 2→1 scenarios).

Baseline and tabular-NN runs live in [`../basemodel-benchmarking/`](../basemodel-benchmarking/).

---

## Model inventory

All DG and DA methods use a **shared MLP backbone** so that family differences reflect the objective, not the architecture (Appendix C.3.1.3–C.3.1.4).

### Domain generalisation (DG) — DomainBed protocols

| Model | Mechanism |
|---|---|
| IRM | Invariant-classifier constraint across source domains. |
| VREx | Risk-variance penalty across source domains. |
| GroupDRO | Worst-group loss (domains as groups). |
| MixStyle | Mix feature statistics across training domains. |
| MLDG | Meta-learning for domain generalisation. |
| MASF | Task generalisation + feature-space alignment + class-metric learning (ported from the original author release). |
| Fish | Gradient-matching meta-updates across domains. |
| CSD | Common-Specific Decomposition of representation. |
| SagNet | Style-Agnostic Network (content / style separation). |

Upstream reference: [DomainBed](https://github.com/facebookresearch/DomainBed) protocols for search space and evaluation.

### Domain adaptation (DA) — Transfer-Learning-Library (TLL) protocols

| Model | Mechanism |
|---|---|
| DANN | Adversarial domain discriminator. |
| CDAN | Conditional adversarial alignment (features + predictions). |
| DAN | Multi-kernel MMD feature alignment. |
| DeepCORAL | Covariance matching (ported from original release). |
| MCC | Minimum class confusion on target predictions. |
| ADDA | Two-stage adversarial adaptation (ported from original release). |
| MCD | Maximum classifier discrepancy on target (ported from original release). |
| JAN | Joint-distribution MMD (features + predictions). |
| SHOT | Source-free target adaptation (ported from original release). |
| CBST | Class-balanced self-training on pseudo-labels (ported from original release). |
| CGDM | Cross-domain gradient-discrepancy minimisation (ported with MLP-style feature extractor + dual classifiers). |

Upstream reference: [Transfer-Learning-Library (TLL)](https://github.com/thuml/Transfer-Learning-Library) for DA protocols and search-space adaptation.

---

## Folder layout

```
domain_adaptation/
├── README.md                # this file
├── __init__.py
└── models/                  # in-repo DG / DA implementations
    ├── __init__.py
    ├── _da_helpers.py       # shared EarlyStopTracker / DataLoader utilities for train_* loops
    ├── da_tllib_losses.py   # TLL-style adversarial / MMD / coral losses shared by the DA family
    ├── da_models.py         # back-compat shim re-exporting from da/*
    ├── domainbed_algos.py   # back-compat shim re-exporting from dg/*
    ├── da/                  # one file per DA algorithm (TLL protocols)
    │   ├── __init__.py
    │   ├── dann.py          # DANN
    │   ├── cdan.py          # CDAN
    │   ├── dan.py           # DAN
    │   ├── deepcoral.py     # DeepCORAL
    │   ├── mcc.py           # MCC
    │   ├── adda.py          # ADDA
    │   ├── mcd.py           # MCD
    │   ├── jan.py           # JAN
    │   ├── shot.py          # SHOT
    │   ├── cbst.py          # CBST
    │   └── cgdm.py          # CGDM
    └── dg/                  # one file per DG algorithm (DomainBed protocols)
        ├── __init__.py
        ├── _base.py         # shared FeatureClassifier / backbone wiring
        ├── _train.py        # shared DG training loop (source-only val selection)
        ├── erm.py           # ERM
        ├── irm.py           # IRM
        ├── vrex.py          # VREx
        ├── gdro.py          # GroupDRO
        ├── mixstyle.py      # MixStyle
        ├── mldg.py          # MLDG
        ├── masf.py          # MASF
        ├── fish.py          # Fish
        ├── csd.py           # CSD
        └── sagnet.py        # SagNet
```

The `da_models.py` and `domainbed_algos.py` modules at the top of `models/` are kept as **backward-compatible shims** — existing imports of the form `from domain_adaptation.models.da_models import DANN, train_dann` continue to work and resolve to the per-algorithm files under `da/` and `dg/`.

---

## Running

The runnable entry points live in [`../basemodel-benchmarking/`](../basemodel-benchmarking/) so that DG / DA models share the same data loader, logger, and HPO loop as the baselines. Run from the repo root, passing a DG / DA model name (and `--uda` for DA):

```bash
# Setting B — within-wave DA
python basemodel-benchmarking/benchmark.py \
    --dataset D-1 --label valence --model DANN --backbone MLP --uda \
    --hpo_trials 30 --hpo_mode fold1

# Setting C — cross-wave DA
python basemodel-benchmarking/cross_dataset.py \
    --label arousal --model CDAN --uda \
    --hpo_trials 30 --run_setting one_to_one
```

See the setting-level READMEs for the protocol:

- Setting B: [`../benchmark/setting_b/README.md`](../benchmark/setting_b/README.md)
- Setting C: [`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md)

Dependencies are pinned in the repo root [`requirements.txt`](../requirements.txt).

## Outputs

Summary tables aggregated from runs are written to [`../benchmark/results/setting_b_dg.csv`](../benchmark/results/setting_b_dg.csv), [`../benchmark/results/setting_b_da.csv`](../benchmark/results/setting_b_da.csv), [`../benchmark/results/setting_c_dg.csv`](../benchmark/results/setting_c_dg.csv), and [`../benchmark/results/setting_c_da.csv`](../benchmark/results/setting_c_da.csv) using the output contract in [`../benchmark/utils/README.md`](../benchmark/utils/README.md).
