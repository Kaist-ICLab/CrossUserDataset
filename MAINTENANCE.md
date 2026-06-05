# Maintenance Plan

This document states the maintenance, versioning, and long-term-support commitments for the **CrossUserDataset** release.

---

## 1. Hosting

| Artefact | Host | Persistent identifier |
|---|---|---|
| Dataset files (D1 / D2 / D3 archives) | Harvard Dataverse | DOI (assigned on publication) |
| Documentation, schema, benchmark scaffolding | GitHub | Repository URL (this repo) |
| Code license | GitHub ([`LICENSE`](./LICENSE), TBD before public release) | — |
| Data license / DUA | Harvard Dataverse ([`LICENSE-DATA.md`](./LICENSE-DATA.md), TBD before public release) | DOI-anchored |

Harvard Dataverse provides:

- Long-term preservation guarantees (institutionally supported).
- Version-specific DOIs on every revision.
- Immutable archival storage with integrity verification.

If Harvard Dataverse becomes unavailable (unlikely), the authors will re-host the dataset on an equivalent open-academic repository (e.g., Zenodo, DataverseNL) and announce the new DOI on this repository's `README.md` and via a pinned GitHub issue.

---

## 2. Maintenance commitment

- **Minimum support horizon:** 3 years from the initial release date.
- **Effort level:** Best-effort issue triage by the author team; community contributions welcome.

---

## 3. Contact and issue response

- **Preferred channel:** GitHub issues on this repository.
- **Backup channel:** email **[TBD contact email on acceptance]**.
- **Service-level expectations:**
  - Acknowledge new issues within **10 business days**.
  - Fix documentation / metadata issues in the **next minor revision** (typically within 30 days).
  - Respond to privacy/ethics reports within **10 business days** (see [`RESPONSIBILITY.md`](./RESPONSIBILITY.md)).
  - Data-quality reports requiring a new Dataverse revision are grouped into periodic releases to keep DOIs stable.

---

## 4. Versioning policy

### 4.1 Semantic versioning for schema and metadata

The repository uses **semantic versioning** (MAJOR.MINOR.PATCH) for its **schema and metadata** artefacts:

- **MAJOR** — breaking changes to column naming, file structure, or label semantics.
- **MINOR** — additive changes (new labels, new participant cohorts, new feature groups) that remain backwards-compatible for existing code.
- **PATCH** — documentation fixes, metadata clarifications, alias-map additions, regenerated summary tables.

Each release is tagged in git and matched to a Dataverse revision.

### 4.2 Dataset revisions (Dataverse)

- Every substantive dataset update publishes a **new Dataverse version** with its own DOI; prior versions remain accessible.
- Corrections that do **not** affect the data (README fixes, documentation improvements, schema clarifications) are published as patch-level GitHub releases without a new Dataverse DOI.

### 4.3 Data-file updates

When a data-level update is required (e.g., a newly discovered preprocessing bug, an additional alias mapping, or a wave-level QC correction):

1. Document the change in [`CHANGELOG.md`](./CHANGELOG.md).
2. Publish a new Dataverse version; keep prior versions live.
3. Update [`metadata/checksums.md5`](./metadata/checksums.md5) and [`metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json`](./metadata/croissant_rai_EmoPhone_A_Multi-Wave_Dataset_for_In-the-Wild_Mob.json) to reference the new files.
4. Tag the corresponding git release (e.g., `v1.1.0`).

---

## 5. Deprecation policy

- **We do not remove** prior dataset revisions. Dataverse retains all versions indefinitely.
- When a revision is superseded, the new revision's landing page will note the supersession and link to the prior version.
- If a file is found to be materially incorrect, the affected revision will be marked **deprecated** rather than deleted.

---

## 6. Future waves

- The D1 / D2 / D3 release is intended as a **stable archival release**.
- Any future collection waves (hypothetical D4, D5, ...) will be released as a **separate Dataverse entry**, cross-linked from this repository, and documented in a new dataset-documentation folder. Such additions do not modify the D1 / D2 / D3 release.

---

## 7. Contribution pathway

External contributions to the code, documentation, or benchmarking scaffolding are welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the PR workflow and review policy. Substantive contributions (e.g., new benchmark setting, new DG/DA method) are reviewed by the maintainer team and may be included in a minor release.

Contributions that affect the dataset content itself (new labels, corrections) must be coordinated with the authors by email.
