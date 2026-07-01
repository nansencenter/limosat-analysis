# LiMOSAT Analysis

Notebooks and  analysis code for working with LiMOSAT drift output data.

This repository is intended for general examples and shared utilities.

## Repository Layout

- `notebooks/` - example notebooks for reading, checking, and visualising LiMOSAT outputs.
- `data/` - local input data used while running notebooks. 
- `outputs/` - generated figures, tables, and intermediate analysis products.
- `src/limosat_analysis/` - future shared Python utilities, added only when code is reusable across notebooks.

## First Example

`notebooks/read_limosat.ipynb` reads a LiMOSAT trajectory table from a Feather file and shows basic point, trajectory, and drift-vector plots.

The notebook expects a local table with these columns:

- `i` - image index
- `g` - trajectory index
- `d` - datetime
- `x`, `y` - projected coordinates in metres
- `q` - quality value

Place the input file at `data/points.feather`, or update `POINTS_PATH` in the notebook.

## Dependencies

The first notebook uses `numpy`, `pandas`, `matplotlib`, `cartopy`, and a Feather reader such as `pyarrow`.
