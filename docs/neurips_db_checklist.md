# NeurIPS Datasets and Benchmarks Track — Artefact Checklist

A reviewer-facing checklist that pre-answers the standard Datasets and Benchmarks Track artefact requirements. Each row is cross-linked to the authoritative source in this repository.

---

## Documentation

| Requirement | Status | Location |
|---|---|---|
| Gebru-style **Datasheet for Datasets** | ✓ | [`../DATASHEET.md`](../DATASHEET.md) |
| **Croissant** ML Commons metadata | ✓ | [`../metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json`](../metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json) |
| Per-wave **schema** / column reference | ✓ | [`../data/schema.md`](../data/schema.md) |
| **Data README** (download, folder structure, loading) | ✓ | [`../data/README.md`](../data/README.md) |
| **Repository README** (overview, benchmark, layout) | ✓ | [`../README.md`](../README.md) |
| **Dataset overview** narrative | ✓ | [`./dataset_overview.md`](./dataset_overview.md) |
| **Feature-alignment** report (cross-wave schema differences) | ✓ | [`./feature_alignment.md`](./feature_alignment.md), mirrored at [`../preprocessing/feature_alignment.md`](../preprocessing/feature_alignment.md) |

## Licensing

| Requirement | Status | Location |
|---|---|---|
| **Code license** clearly stated | TBD on acceptance/public release | [`../LICENSE`](../LICENSE) |
| **Data license** clearly stated | TBD on acceptance/public release | [`../LICENSE-DATA.md`](../LICENSE-DATA.md) |
| License applicability distinguished (code vs data) | ✓ | Code and data placeholders live in separate files |

## Hosting and persistent identifier

| Requirement | Status | Location |
|---|---|---|
| Data hosted on a stable public repository | ✓ (Harvard Dataverse) | [`../README.md § Data Access`](../README.md#data-access); DOI TBD on acceptance |
| **Persistent DOI** committed to | ✓ (placeholder `TBD on acceptance`) | [`../CITATION.cff`](../CITATION.cff), [`../metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json`](../metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json), [`../README.md`](../README.md) |
| Integrity verification | ✓ (checksum file placeholder) | [`../metadata/checksums.md5`](../metadata/checksums.md5) |

## Citation

| Requirement | Status | Location |
|---|---|---|
| Machine-readable **CITATION.cff** | ✓ | [`../CITATION.cff`](../CITATION.cff) |
| Bibtex entry in README | ✓ | [`../README.md § Citation`](../README.md#citation) |
| Upstream pipeline citation (Zhang et al., 2024) | ✓ | README, DATASHEET, CITATION.cff, AUTHORS.md |

## Ethics and privacy

| Requirement | Status | Location |
|---|---|---|
| **IRB approval** documented | ✓ (number TBD on acceptance) | [`../DATASHEET.md § 3`](../DATASHEET.md), [`./ethics.md`](./ethics.md) |
| **Informed consent** documented | ✓ | [`../DATASHEET.md § 3`](../DATASHEET.md), [`./ethics.md`](./ethics.md), translated form placeholder at [`./consent_form_en.md`](./consent_form_en.md) |
| **Anonymisation** procedures documented | ✓ | [`../DATASHEET.md § 2`](../DATASHEET.md), [`./ethics.md § 2`](./ethics.md) |
| **Risk register** | ✓ | [`./ethics.md § 4`](./ethics.md) |
| **Broader impact** discussion | ✓ | [`./ethics.md § 5`](./ethics.md), paper Section 7 |

## Author responsibility and maintenance

| Requirement | Status | Location |
|---|---|---|
| **Author responsibility statement** ("we bear all responsibility...") | ✓ | [`../RESPONSIBILITY.md`](../RESPONSIBILITY.md) |
| **Maintenance plan** with SLA and versioning | ✓ (≥ 3-year commitment) | [`../MAINTENANCE.md`](../MAINTENANCE.md) |
| **Changelog** | ✓ | [`../CHANGELOG.md`](../CHANGELOG.md) |
| Contribution workflow | ✓ | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Authorship and acknowledgements | ✓ (author list TBD on acceptance) | [`../AUTHORS.md`](../AUTHORS.md) |

## Reproducibility

| Requirement | Status | Location |
|---|---|---|
| **Benchmark specification** (splits, filters, metrics) | ✓ (three settings) | [`../benchmark/README.md`](../benchmark/README.md), [`../benchmark/setting_a/README.md`](../benchmark/setting_a/README.md), [`../benchmark/setting_b/README.md`](../benchmark/setting_b/README.md), [`../benchmark/setting_c/README.md`](../benchmark/setting_c/README.md) |
| **Model inventory** with upstream provenance | ✓ | [`../benchmark/README.md § Model inventory`](../benchmark/README.md), [`../domain_adaptation/README.md`](../domain_adaptation/README.md), [`../basemodel-benchmarking/README.md`](../basemodel-benchmarking/README.md) |
| **Preprocessing decisions** (QC, scale normalisation, alias map) | ✓ | [`../preprocessing/pipeline_decisions.md`](../preprocessing/pipeline_decisions.md) |
| **Loader / metric contract** | ✓ | [`../benchmark/utils/README.md`](../benchmark/utils/README.md) |
| **Seed and HPO policy** documented | ✓ (fixed seed, Optuna 30 trials, val-AUROC selection) | `../benchmark/README.md`, `../benchmark/utils/README.md` |
| **Dependencies pinned** | ✓ | [`../requirements.txt`](../requirements.txt) |

## Dataset accessibility

| Requirement | Status | Location |
|---|---|---|
| Free access for academic research | ✓ (DUA, immediate access) | [`../README.md § Data Access`](../README.md#data-access) |
| Download instructions (GUI + CLI + Python) | ✓ | [`../data/README.md § Download`](../data/README.md) |
| Gated / DUA-protected distribution | ✓ | [`../LICENSE-DATA.md`](../LICENSE-DATA.md) |
| Structured loading examples | ✓ | [`../data/README.md § Loading Examples`](../data/README.md) |

---

## Items held as `[TBD on acceptance]`

The following placeholders will be populated on paper acceptance (they are held during peer review for anonymity):

- Author names and affiliations ([`../AUTHORS.md`](../AUTHORS.md), [`../CITATION.cff`](../CITATION.cff), [`../DATASHEET.md`](../DATASHEET.md)).
- IRB number and institution ([`../DATASHEET.md`](../DATASHEET.md), [`../RESPONSIBILITY.md`](../RESPONSIBILITY.md), [`./ethics.md`](./ethics.md)).
- Harvard Dataverse DOI and file IDs ([`../metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json`](../metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json), [`../CITATION.cff`](../CITATION.cff), [`../README.md`](../README.md)).
- Contact email ([`../README.md`](../README.md), [`../DATASHEET.md`](../DATASHEET.md), [`../MAINTENANCE.md`](../MAINTENANCE.md), [`../CONTRIBUTING.md`](../CONTRIBUTING.md), [`../RESPONSIBILITY.md`](../RESPONSIBILITY.md)).
- Code and data licenses / DUA terms ([`../LICENSE`](../LICENSE), [`../LICENSE-DATA.md`](../LICENSE-DATA.md), [`../CITATION.cff`](../CITATION.cff)).
- Full translated consent form ([`./consent_form_en.md`](./consent_form_en.md)).
- Committed benchmark result CSVs (headers present; rows populated once production runs are committed; see [`../benchmark/results/README.md`](../benchmark/results/README.md)).
- SHA256 / MD5 checksums for the Dataverse archives ([`../metadata/checksums.md5`](../metadata/checksums.md5)).
