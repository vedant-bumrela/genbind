# MoleGen - SELFIES VAE molecule generator

## Quickstart
1. Create a virtualenv and install requirements:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

2. Add raw SMILES:
   Put a file `data/raw/smiles_raw.txt` containing one SMILES per line.
   For fast testing, you can copy 200–500 SMILES lines from any small source (ZINC subset, ChEMBL snippets).

3. Prepare:
   bash scripts/run_prep.sh

4. Train:
   bash scripts/run_train.sh

5. Sample:
   bash scripts/run_sample.sh

6. Evaluate:
   python src/evaluate.py --samples_path outputs/samples.txt --processed_path data/processed/smiles_train.txt

## What to do today (priority)
1. Get a small SMILES file (200–2000 lines). Put it in `data/raw/smiles_raw.txt`.
2. Run prep -> creates `data/processed/smiles_train.txt`.
3. Run train (20 epochs default). Watch losses & checkpoints in outputs/checkpoints.
4. Run sample and evaluate metrics.
5. If QC OK, scale dataset and epochs.

## Next improvements
- Replace VAE with Transformer encoder/decoder.
- Use SELFIES token-level conditioning for properties.
- Train property predictors (QED, logP, SA) and run RL/Bayesian optimization on top.
