# Conv-LBI review fixes — 2026-08-26

These files are replacements relative to the project root `260817_iclr2027-refined/`.

Replaced files:

```text
core/lbi/groups.py
core/lbi/engine.py
core/lbi/diagnostics.py
shot_otta/trainer.py
shot_otta/config.py
experiment_identity.py
tools/plan_experiments.py
tools/summarize_runs.py
```

Recommended replacement:

```bash
cd /path/to/260817_iclr2027-refined
cp -a core/lbi/groups.py core/lbi/groups.py.before_conv_review
cp -a core/lbi/engine.py core/lbi/engine.py.before_conv_review
cp -a core/lbi/diagnostics.py core/lbi/diagnostics.py.before_conv_review
cp -a shot_otta/trainer.py shot_otta/trainer.py.before_conv_review
cp -a shot_otta/config.py shot_otta/config.py.before_conv_review
cp -a experiment_identity.py experiment_identity.py.before_conv_review
cp -a tools/plan_experiments.py tools/plan_experiments.py.before_conv_review
cp -a tools/summarize_runs.py tools/summarize_runs.py.before_conv_review

tar -xzf /path/to/CONV_LBI_REVIEW_FIXES_20260826.tar.gz -C .
```

The archive contains project-relative paths, so extracting it at the project root overwrites only the files above plus this README.
