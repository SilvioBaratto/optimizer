# Cycle 0 Baseline

Pre-migration output artifacts for bit-exact parity testing (issue #688).

## Generating the baseline

Run the capture script from the repo root:

```bash
MPLBACKEND=Agg PYTHONHASHSEED=0 python tests/research/baselines/cycle0/capture_baseline.py
```

This runs `research.cli.main` with:
- `--seed 42`
- `--start-date 2022-01-01`
- `--end-date 2023-12-31`

Outputs are written here (`tests/research/baselines/cycle0/`).

## Exit code

The pipeline exit code (0 = all 17/17 rules pass, 1 = any rule fails) is stored
in `.exit_code` so the parity test can verify it matches.

## Regenerating after intentional changes

If production output intentionally changes (algorithm update, new rule, etc.),
delete all artifacts here, re-run `capture_baseline.py`, and commit the new baseline.
