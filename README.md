# DPFairFormer

Official implementation for **DPFairFormer**, a polarity-aware signed graph learning model for mitigating degree-polarity bias in signed link sign prediction.

## Requirements

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
dpfair/
  data.py                 # signed graph loading, split, degree groups, PAPE cache
  layers.py               # polarity-aware transfer and sparse Transformer layers
  models.py               # DPFairSGNN and DPFairFormer
  metrics.py              # AUC, F1, accuracy, DeltaDPSP
  train.py                # training entry
  validate.py             # validation entry for checkpoints
  test.py                 # test entry for checkpoints
  configs/                # example configs
data/
  README.md
  Bitcoinalpha.txt
  Bitcoinotc.txt
  WikiRfa.txt
  Slashdot.txt
  amazon_book.txt
```

Each dataset file is a whitespace- or comma-separated edge list:

```text
src dst sign_or_rating
```

## Quick Start

Run DPFairFormer on Bitcoin-OTC:

```bash
python -m dpfair.train --config dpfair/configs/dpfairformer/bitcoin_otc.yaml
```

Run with command-line arguments:

```bash
python -m dpfair.train --dataset Bitcoinotc --model dpfairformer --epochs 200 --pape_hops 2
python -m dpfair.train --dataset Bitcoinotc --model dpfairsgnn --epochs 200
```

Training creates a timestamped run directory under `dpfair/runs/` with:

- `args.json`: run configuration
- `metrics.csv`: per-epoch validation metrics
- `best.pt`: checkpoint selected by the configured validation criterion
- `best_auc.pt`, `best_dpsp.pt`, `best_tradeoff.pt`: metric-specific checkpoints
- `summary.json`: best validation metrics and final test metrics

## Evaluation

```bash
python -m dpfair.validate --checkpoint dpfair/runs/<run_name>/best.pt
python -m dpfair.test --checkpoint dpfair/runs/<run_name>/best.pt
```

Reported metrics include AUC, F1, macro-F1, accuracy, and DeltaDPSP. DeltaDPSP measures the signed prediction-rate gap between degree-polarity edge groups.


## Citation

If this repository is useful for your research, please cite the DPFairFormer paper.
