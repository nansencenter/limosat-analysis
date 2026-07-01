# LiMOSAT Analysis

Notebooks and  analysis code for working with LiMOSAT drift output data.

This repository is intended for general examples and shared utilities.

## Repository Layout

- `notebooks/` - example notebooks for reading, checking, and visualising LiMOSAT outputs.
- `data/` - local input data used while running notebooks. 
- `outputs/` - generated figures, tables, and intermediate analysis products.
- `src/limosat_analysis/` - future shared Python utilities, added only when code is reusable across notebooks.

## Examples

`notebooks/read_limosat.ipynb` reads a LiMOSAT trajectory table from a Feather file and shows basic point, trajectory, and drift-vector plots.

`notebooks/grid_limosat.ipynb` shows how to derive segment velocities from the trajectory table and aggregate them to a 25 km grid.

`notebooks/select_limosat_trajectories.ipynb` shows how to select all trajectories passing through a chosen space-time domain.

The notebooks expect a local table with these columns:

- `i` - image index
- `g` - trajectory index
- `d` - datetime
- `x`, `y` - projected coordinates in metres
- `q` - quality value

Place the input file at `data/points.feather`, or update `POINTS_PATH` in the notebook.

## Dependencies

The notebooks use `numpy`, `pandas`, `matplotlib`, `cartopy`, `pyproj`, and a Feather reader such as `pyarrow`.
