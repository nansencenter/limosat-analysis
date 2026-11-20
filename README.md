# LiMOSAT Analysis

A reader and example notebooks for the public LiMOSAT trajectory release: Lagrangian sea-ice
trajectories from Sentinel-1, May 2015 to July 2025.

## The release

The release is one NetCDF4 file per processing period, `limosat_<period>.nc`, with a `manifest.json` and a
`README.md` (the data dictionary, attribution and licence) beside them. The files follow the CF conventions
(featureType `trajectory`, contiguous ragged arrays) and describe themselves, so `ncdump -h`, xarray, MATLAB, R
and any other NetCDF tool can read them. Every column is its own variable, so a reader that wants only a few of
them decompresses only those. This package adds trajectory-aware access on top (date and space-time windows,
whole trajectories, displacements with both ends, flag filters, checksum verification).

A row is one **observation** of a feature in one Sentinel-1 scene. The rows of a trajectory share
`trajectory_id` and are ordered by `point_index` (0 at the first observation), so a trajectory is read as an
unbroken track by sorting on `point_index`. Every row after the first of a trajectory also describes the
**displacement** from the previous observation. Rows are sorted by time.

| Column | Meaning |
| --- | --- |
| `period`, `trajectory_id`, `point_index` | identity. `trajectory_id` is unique only within a period |
| `time`, `scene` | UTC acquisition time and Sentinel-1 product name of the scene |
| `x_m`, `y_m` | position, EPSG:3413 metres |
| `dx_m`, `dy_m`, `speed_m_per_day` | displacement and speed from the previous observation (null on the first row) |
| `interpolated`, `correlation`, `direct_match_share`, `quality_flags` | properties of that displacement |
| `is_primary` | the row lies in the period's own date window |

Processing periods are 15 months long and overlap their neighbours by three months, and all rows of every
period are published. `is_primary` marks the rows in a period's own window (1 August to the next 1 August;
the first period also owns its first three months). Keeping `is_primary` rows from all periods covers each
date exactly once. `quality_flags` is a bit mask of informational flags that never split trajectories; the
names and bits are in `data.metadata["flag_bits"]`. See the release's `README.md` for the full definitions.

## Installation

Use your existing conda/mamba environment, then install this repository from its root into that
environment:

```bash
python -m pip install -e ".[notebooks]"
```

The reader requires Python 3.10+, NumPy, pandas and netCDF4. The notebook extra adds Matplotlib, Cartopy,
pyproj and a Jupyter kernel. For the reader alone, use `python -m pip install -e .`.

## Reading a period

```python
from limosat_analysis import open_limosat

data = open_limosat("limosat_202305_202407.nc")
data.metadata          # schema_version, release_id, period, crs, flag_bits, primary_window, rows
data.describe()        # column, type, unit and description, as recorded in the file
data.verify()          # SHA-256 against manifest.json in the same folder

# Observations in a date range, in the period's own window. By default a read returns the core columns:
# trajectory_id, point_index, time, x_m, y_m, dx_m, dy_m, speed_m_per_day, is_primary
points = data.read_points(start="2024-01-01", stop="2024-02-01", primary_only=True)

# Choose columns: a list of names, or "all" (adds scene, lon, lat, interpolated, correlation, ...)
more = data.read_points(start="2024-01-01", stop="2024-02-01", columns=["time", "lon", "lat", "correlation"])
everything = data.read_points(start="2024-01-01", stop="2024-01-08", columns="all")

# Complete trajectories that have an observation inside a space-time window
tracks = data.read_trajectories(start="2024-01-01", stop="2024-01-08",
                                xlim=(0, 1_000_000), ylim=(-1_000_000, 0))
for trajectory_id, track in tracks.groupby("trajectory_id"):
    ...                # one trajectory at a time, ordered by point_index

# One row per displacement, with both ends; drop displacements by flag name
moves = data.read_displacements(start="2024-01-01", stop="2024-02-01",
                                exclude_flags=["MASK_FALSE_LAND"], primary_only=True)
```

| Method | Returns |
| --- | --- |
| `read_points(start, stop, trajectory_ids, columns, primary_only, max_rows)` | observations, sorted by time; core columns unless `columns` is a list or `"all"` |
| `read_trajectories(start, stop, xlim, ylim, trajectory_ids, columns, max_rows)` | every observation of each trajectory that has one in the window, ordered by `trajectory_id` and `point_index` |
| `read_displacements(start, stop, trajectory_ids, columns, primary_only, exclude_flags, max_rows)` | one row per displacement: `trajectory_id`, `point_index` (of the end), `start_time`, `end_time`, `elapsed_seconds`, `x0_m`, `y0_m`, `x1_m`, `y1_m`, the displacement columns, `start_scene`, `end_scene`, `is_primary`, `lon`, `lat`; core columns by default |
| `describe()`, `verify()` | the column dictionary recorded in the file; a checksum check against `manifest.json` |

- Dates are UTC and half-open: `start <= t < stop`. Naive dates are UTC. For `read_points` and
  `read_trajectories` they select observation times; for `read_displacements` they select the time the
  displacement starts, which is also what `is_primary` is defined on. `elapsed_seconds` is derived from the two
  times (it is not stored in the files).
- Flag names come from the file, so a new flag in a later release needs no code change.
  `exclude_flags` accepts the names in `data.metadata["flag_bits"]`.
- Results are pandas DataFrames held in memory. In the NetCDF files trajectories are stored in start-time order,
  so a date window and a single trajectory are a few contiguous reads; a whole period with all columns is about a
  gigabyte of decompression (seconds), so ask for the columns and dates you need. `max_rows` is a preview limit,
  not a sample.
- The core columns are `CORE_COLUMNS` and `CORE_DISPLACEMENT_COLUMNS` (importable from `limosat_analysis`).
- The reader never modifies the files, and does not reproject coordinates.
- Periods are not combined automatically: open the files you need and keep `is_primary` rows (or use
  `period` plus `trajectory_id`) when stacking them.

## Notebooks

Set `DATA_PATH` in the first cell to a period file of the release; choose UTC dates. The defaults read
`../data/limosat_202305_202407.nc`: link or copy the release files there. No data are distributed with
this repository.

- `notebooks/read_limosat.ipynb`: points, trajectories, drift vectors and quality flags for a date range.
- `notebooks/grid_limosat.ipynb`: aggregate projected vector components from displacements onto a daily
  grid, optionally excluding flagged displacements.
- `notebooks/select_limosat_trajectories.ipynb`: select complete trajectories through a space-time window.

These examples plot EPSG:3413 Arctic data.

## Older sources

This version reads the public release only. Raw tracking databases, the earlier benchmark Feather files
(`pandas.read_feather` reads them directly) and the internal QC v1 archive (kept-link Parquet, removed links
and SQLite) are no longer read by this package. They remain available in the earlier versions of this
repository, up to commit `6d3694b` (and branch `mask-false-land-flag`).

## Development checks

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The tests build a small synthetic NetCDF file in the release layout and cover opening and metadata, time and
trajectory filters, complete trajectories through a space-time window, displacements and flags, version and
column checks, checksum verification and rejection of other sources. Local inputs and outputs belong in
ignored `data/` and `outputs/` directories.
