# Soft-DTW Counterfactual Explanations (ACIIDS 2025 submission)

- Repository for the paper *Towards plausibility in time series counterfactual
explanations* submitted to ACIIDS 2025.
- Method: gradient-based optimization in input space with plausibility enforced via soft-DTW alignment to k-NN target-class exemplars; losses for validity, sparsity, and proximity to keep changes minimal and localized. See abstract above for the research summary.
- Baseline/reference methods live in `src/soft_dtw_cfe/reference/m_cels` and `src/soft_dtw_cfe/reference/glacier`. Datasets are fetched automatically from aeon into `data/`.

## Environment (uv)
- Fast path: `./scripts/create_uv_env.sh` (creates `.venv` and installs from `pyproject.toml` / `uv.lock`).
- Manual: `uv venv && source .venv/bin/activate && uv sync`.
- Use `uv run ...` for all commands below to stay inside the uv environment.

## Reproduce paper results (order matters)
1) Train classifiers (Optuna)
   - Command:  
     `uv run python src/soft_dtw_cfe/experiments/train_classifier_optuna.py --datasets CBF TwoLeadECG GunPoint Earthquakes Coffee ItalyPowerDemand Cricket Epilepsy --classifier-trials 30 --clf-epochs 80 --output-dir optuna_experimentsv2`
   - Outputs per dataset: `optuna_experimentsv2/<dataset>_optuna/checkpoints/best_classifier.pt`, trial history CSV, and metadata.
   - Reuse checkpoints with `--skip-if-checkpoint`; skip Optuna with `--skip-clf-optim`; force CPU with `--force-cpu`.

2) Evaluate Soft-DTW counterfactual solver
   - Command:  
     `uv run python src/soft_dtw_cfe/experiments/evaluate_solver.py --datasets CBF TwoLeadECG GunPoint Earthquakes Coffee ItalyPowerDemand Cricket Epilepsy --steps 300 --lambda-validity 1 --k-neighbors 10 --output-dir optuna_experimentsv2`
   - Outputs per dataset under `<dataset>_optuna`: `results/evaluation_metrics.json`, `results/counterfactuals.npz`, `visualizations/counterfactual_examples.png`.

3) (Optional) Run hyperparams search
   - Command:  
     `uv run python src/soft_dtw_cfe/experiments/hyperparams.py --datasets CBF TwoLeadECG GunPoint --lambdas 1 2 5 --k-neighbors-list 5 10 20 --n-samples 50 --optuna-dir optuna_experimentsv2`
   - Produces per-dataset hyperparametrs search taking into account plausability, proximity, validity and execution time.

## Notes
- Defaults used in the paper: `seed=42`, `test_size=0.2`; keep these to match reported results.
- The repository already contains `optuna_experimentsv2` with previous runs; rerun the commands above to regenerate them.
- Reference baselines (m-cels, glacier) can be run from their respective subfolders in `src/soft_dtw_cfe/reference`.

## Reference methods
- Glacier baseline: `uv run python src/soft_dtw_cfe/reference/glacier/evaluate_glacier.py --use-saved-classifier --optuna-dir optuna_experimentsv2`  
  (script currently runs the aeon datasets listed inside the file; edit the `UNI_DATASETS` / `MULTI_DATASETS` lists there to change the set).
- M-CELS baseline: `uv run python src/soft_dtw_cfe/reference/m_cels/evaluate_mcels.py --datasets CBF TwoLeadECG --optuna-dir optuna_experimentsv2`  
  (reuses Optuna checkpoints; pass your aeon dataset names via `--datasets`).
