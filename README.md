# Structure-Function Synergy in Graph Neural Networks

This supplementary package contains the code for the vanilla and PGH-based GNN experiments reported in the paper.

## Environment

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate reproduction
```

## Running experiments

List all available experiment presets:

```bash
python main.py --list-presets
```

Example for a TUDataset experiment:

```bash
python main.py --preset tu_gin_pgh_1l --dataset MUTAG
```

Example for an LRGB Peptides experiment:

```bash
python main.py --preset peptides_func_gine_pgh
```

Supported TUDatasets include:

```text
MUTAG
ENZYMES
PROTEINS
COLLAB
REDDIT-BINARY
IMDB-BINARY
```

Datasets are downloaded automatically through PyTorch Geometric. Experimental results are saved under:

```text
outputs/results/
```

