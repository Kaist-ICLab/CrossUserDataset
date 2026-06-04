# Preprocessing & Feature Extraction Pipeline (D-2, D-3, D-4)

End-to-end pipeline that turns raw mobile-sensing logs from the **D#2 / D#3 / D#4** datasets into the **stress feature pickle** (`stress_binary_personal-full.pkl`) used downstream by the Overfitting study.

> **Scope.** This folder is *ESM-synchronized stress only*. The hourly-interpretable pipeline and the objective-label notebooks (`Label_Extraction_ESMsync_step*.ipynb`, `Label_Extraction_ESMsync_screen.ipynb`, `Label_Extraction_Hourly.ipynb`) are intentionally **excluded** and remain in the original `D-2/`, `D-3/`, `D-4/` directories.

> **⚠️ Label-binarization caveat.** The `stress_binary_personal-full.pkl` produced here is **only valid for Tier B and Tier C experimental settings**. In both tiers the personal binarization threshold is computed from *each user's full ESM response distribution*, which is exactly what `Label_Extraction_ESMsync_*` produces.
>
> **Tier A (temporal/chronological split) requires a different pipeline.** Because Tier A reserves each user's earliest responses as the training set, the per-user threshold must be derived from the *training subset only* — not the full distribution. Do **not** reuse `stress_binary_personal-full.pkl` for Tier A; rebuild labels per-user from each user's training fold.

---

## Privacy & Anonymization

This release has been scrubbed of **participant-level PII** and **personal credentials** while leaving research-relevant configuration (e.g. campus geofences) intact — the manuscript is single-blind, so institution-identifying constants are fine.

| Concern | Action taken |
| --- | --- |
| Participant IDs (`P001`, `P002`, …) embedded in notebook cell outputs | All notebook outputs **fully cleared** (`jupyter nbconvert --clear-output`) |
| Per-participant valid-user allow-list (`valid_users.txt`) | **Deleted** along with its two helper scripts (`load_valid_users.py`, `find_problematic_user.py`) — they were only used by the hourly pipeline anyway |
| Hardcoded absolute paths (`/var/nfs_share/D#X`, `/var/nfs_share/Overfitting/D-X`) | Driven by `DATA_ROOT` / `PROJECT_ROOT` env vars in `Funcs/D-*/Utility.py`, with `<DATA_ROOT>` / `<PROJECT_ROOT>` defaults; notebook driver paths replaced with `<PROJECT_ROOT_D{2,3,4}>` placeholders |
| Personal API keys / tokens | None present — the only key reference is the `API_KEY = "YOUR_API_KEY"` placeholder for `googlemaps.Client`. (The real Google Maps key found in the originals lived in `Preprocessing_hourly.ipynb`, which is excluded from this release.) |
| Study-site geofence coordinates (KAIST main / Munji / Seoul) | **Kept as-is** — public addresses, single-blind submission |

Before publishing, please re-scan your changes:

```bash
grep -rE "(P0[0-9]{3}|AIzaSy|sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]+)" Preprocessing/
```

---

## Folder Layout

```
Preprocessing/
├── README.md
├── common/
│   └── Funcs/                              # Shared Python helpers (per dataset; see Note 1)
│       ├── D-2/{Utility.py, Extract_sliding_help.py, __init__.py}
│       ├── D-3/{Utility.py, Extract_sliding_help.py, __init__.py}
│       └── D-4/{Utility.py, Extract_sliding_help.py, __init__.py}
│
├── 01_preprocessing/                       # Stage 1 — raw CSV → cleaned intermediates
│   ├── app_categorization/
│   │   ├── AppCategorization_D{2,3,4}.{py,ipynb}    # Selenium scraper: package → Play-Store category
│   │   └── update_location_data_D2.py               # D-2 only: location data patch
│   └── esm_preprocessing/
│       └── Preprocessing_ESM_D{2,3,4}.ipynb         # Sensor → per-modality intermediate; ESM cleaning
│
└── 02_feature_extraction/                  # Stage 2 — intermediates → features + labels
    ├── Label_Extraction_ESMsync_D{2,3,4}.ipynb      # Subjective ESM label preparation
    └── Feature_Extraction_D{2,3,4}.ipynb            # Sliding-window features around each ESM prompt
                                                     # ▶ stress_binary_personal-full.pkl
```

**Note 1 — why `Funcs/` is per-dataset.** `Utility.py` is byte-identical across datasets except for path constants; `Extract_sliding_help.py` is identical across all three. Rather than refactor every notebook to read paths from a config, we keep the per-dataset variants so the notebooks import them unchanged.

---

## Pipeline Flow

```
Raw sensor data (CSV per participant)                    ESM responses (CSV)
        │                                                       │
        ▼                                                       ▼
┌─────────────────────────────────────┐         ┌─────────────────────────────┐
│ 01_preprocessing/app_categorization │         │ 01_preprocessing/           │
│  (Play-Store scrape → categories)   │         │ esm_preprocessing           │
└──────────────────┬──────────────────┘         │ (POI clustering, ESM clean, │
                   │                            │  per-modality processors)   │
                   ▼                            └──────────────┬──────────────┘
        ┌────────────────────────────────────────────┐         │
        │ proc_updated_ESM.pkl  (merged sensor dict) │ ◀───────┘
        └────────────────────────────┬───────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────┐
        │ 02_feature_extraction/                     │
        │  Label_Extraction_ESMsync                  │
        │   → labels_1h_esmsyn.csv                   │
        │  Feature_Extraction (Ray, sliding window)  │
        │   → stress_binary_personal-full.pkl        │
        └────────────────────────────────────────────┘
```

### Stage 1 — Preprocessing (`01_preprocessing/`)

1. **App categorization** (`AppCategorization_D{2,3,4}.{py,ipynb}`)
   - Reads `AppUsageEvent.csv` / `KeyEvent.csv` per participant.
   - Selenium + Firefox scrapes Play Store for each package → assigns a category (`GAME`, `SOCIAL`, `WORK`, …).
   - These raw categories are later collapsed into 6 high-level buckets (`ENTER / INFO / SOCIAL / WORK / HEALTH / SYSTEM / UNKNOWN`) via the `transform` map in `Funcs/Utility.py`.

2. **ESM preprocessing** (`Preprocessing_ESM_D{2,3,4}.ipynb`)
   - Per-modality processors (`_proc_activity_event`, `_proc_screen`, `_proc_call`, `_proc_app_usage`, `_proc_location`, …) clean each raw sensor stream.
   - POI clustering (`PoiCluster`) labels location clusters as `home` / `work` / `none` / `social` / `eating` / `gym`. A three-site geofence (`site1/2/3`) re-tags clusters near study-fixed locations — **set the lat/lon for your study before running** (see `TODO` comments).
   - Writes `proc_updated_ESM.pkl` (a `{modality_key: pd.Series}` dict indexed by `(pcode, timestamp)`).

### Stage 2 — Feature Extraction (`02_feature_extraction/`)

1. **`Label_Extraction_ESMsync_D{2,3,4}.ipynb`**
   - Reads `EsmResponse.csv`, derives binarized stress targets per participant (`stress_binary_personal`).
   - Writes `labels_1h_esmsyn.csv` indexed by `(pcode, timestamp)`.

2. **`Feature_Extraction_D{2,3,4}.ipynb`**
   - Loads `proc_updated_ESM.pkl` + `labels_1h_esmsyn.csv`.
   - For every ESM prompt, runs the sliding-window extractor (`Funcs/Extract_sliding_help.py`) in parallel with Ray.
   - Produces feature statistics (`#AVG`, `#STD`, `#SKW`, `#KUR`, `#ASC`, `#BEP`, `#MED`, `#TSC` for numeric; `#ETP#`, `#ASC#`, `#RLV_SUP` for categorical) plus participant-info one-hots (`PIF#*`).
   - Final output:
     ```
     {PATH_INTERMEDIATE}/stress_binary_personal-full.pkl
     ```
     Pickle stores the tuple `(X, y, group, t_norm, date_times)`. A reduced `stress_binary_personal-current.pkl` is also produced by selecting columns matching `PIF`, `Sleep`, `#VAL`.

---

## Configuration

### 1. Paths

Each `Funcs/D-{2,3,4}/Utility.py` reads two environment variables:

```bash
# Per dataset, point these at your local copy
export DATA_ROOT=/path/to/dataset_D3        # has SubjData/EsmResponse.csv, newdata/<pcode>/*.csv
export PROJECT_ROOT=/path/to/workspace_D3   # Intermediate/ and Results/ are written here
```

Notebook driver paths (the `sys.path.append("/var/nfs_share/Overfitting/D-X")` calls inside notebooks for Ray's `working_dir`) appear as `<PROJECT_ROOT_D2>` / `<PROJECT_ROOT_D3>` / `<PROJECT_ROOT_D4>` placeholders — replace them with your local absolute path before running.

### 2. Study-site geofence (`Preprocessing_ESM_D*.ipynb`)

The original KAIST main / Munji / Seoul coordinates are kept as the default geofence centers used to re-tag location clusters as `work`:

```python
center_lat_kaist, center_lon_kaist = (36.3722, 127.3600); _radius_kaist = 1000  # m
center_lat_munji, center_lon_munji = (36.391944, 127.398611); _radius_munji = 400
center_lat_seoul, center_lon_seoul = (37.5933, 127.0464); _radius_seoul = 300
```

Replace with your own study sites, or drop the `in_circle_*` branches in `_proc_location` if your study has no fixed sites.

### 3. Google Maps API

`Preprocessing_ESM_D*.ipynb` uses `googlemaps.Client(API_KEY)` for `places_nearby` cluster labelling (`label_cluster`). The placeholder string `"YOUR_API_KEY"` is left in source — supply your own key or skip that pass.

---

## Dependencies

Python 3.9+ with:

```
pandas, numpy, scipy, scikit-learn
cloudpickle, ray, dask
xgboost
matplotlib, seaborn
pytz, googlemaps
selenium, beautifulsoup4   # AppCategorization web scrape
poi                        # POI clustering used in Preprocessing_ESM
```

System: **Firefox + geckodriver** for the Selenium scrape (`AppCategorization_D3.py` includes binary auto-discovery; D-2 / D-4 expect `firefox` on `PATH`).

---

## Reproducing the Final Pickle

For each dataset (example: **D-3**):

```bash
export DATA_ROOT=/path/to/dataset_D3
export PROJECT_ROOT=/path/to/workspace_D3

# 1. App categorization (slow — Play-Store scrape)
python 01_preprocessing/app_categorization/AppCategorization_D3.py

# 2. Sensor/ESM preprocessing
jupyter nbconvert --to notebook --execute \
  01_preprocessing/esm_preprocessing/Preprocessing_ESM_D3.ipynb

# 3. Subjective ESM labels
jupyter nbconvert --to notebook --execute \
  02_feature_extraction/Label_Extraction_ESMsync_D3.ipynb

# 4. Final pickle
jupyter nbconvert --to notebook --execute \
  02_feature_extraction/Feature_Extraction_D3.ipynb

# Output:
#   ${PROJECT_ROOT}/Intermediate/stress_binary_personal-full.pkl
```

Identical sequence for D-2 and D-4 with the matching suffix.
