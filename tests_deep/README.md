# Deep-learning tests

Tests that need PyTorch (the `deep` dependency group). Run them **in a separate pytest process**
from `tests/`:

```bash
uv run pytest                                # tests/ (default testpaths)
uv run --group deep pytest tests_deep        # this folder
```

Why separate: PyTorch and LightGBM each bundle their own OpenMP runtime (`libomp`). Loaded into
the same Python process on macOS, they crash with a segmentation fault or hang. The project code
never loads both: `run_cv` imports LightGBM only for `--model lgbm` and PyTorch only for
`--model cnn`. A single pytest run over both folders would load both, because pytest imports
every test file while collecting tests, even ones it then skips.
