# ROGII Wellbore Geology Prediction

Working repository for the [Rogii Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/) Kaggle competition (Featured, RMSE on per-foot TVT along horizontal wellbores, final deadline 2026-08-05).

The repository combines two workstreams:

1. **Competition intelligence** — a mirrored wiki of the competition (discussions, leaderboard, notebooks) built with `wikikit`, plus analysis artifacts in `analysis/` (notebook corpus clustering, strategy archetypes, fork-family maps).
2. **Own solution** — code in `src/` built from those learnings, deliberately decorrelated from the public-notebook fork lineage.

## Orientation

| Where | What |
| --- | --- |
| `docs/ROADMAP.md` | Strategy, phases, and links to all issues |
| `docs/issues/` | One file per work item, with status |
| `docs/data-notes.md` | Verified dataset facts (schema, eval-zone geometry, quirks) |
| `docs/experiments.md` | Append-only experiment ledger (CV/LB per run) |
| `submissions/log.md` | Every Kaggle submission + the mechanics of the path |
| `src/` | Data layer, CV harness, models |
| `scripts/` | Entry points: `run_cv.py`, `eda_sanity.py`, `make_kernel.py` |

## Quick start

```bash
uv sync                                             # installs src/ editable
uv run kaggle competitions download rogii-wellbore-geology-prediction -p data/
cd data && unzip -q rogii-wellbore-geology-prediction.zip && cd ..
uv run python scripts/eda_sanity.py                 # dataset sanity checks
uv run python scripts/run_cv.py --model anchor      # baseline CV run
```

Status (2026-07-16): Phase 0 complete — data layer, calibrated CV harness (CV↔LB gap +0.05 on the first submission, LB 15.718), three baselines. Current work: Phase 1, the GR↔typewell tracker (`docs/issues/004-particle-filter-tracker.md`).
