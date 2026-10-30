# LiMOSAT Analysis

Shared readers and example notebooks for LiMOSAT observations and ice-drift
analysis. The LiMOSAT QC v1 release (kept-link Parquet and QC SQLite) is the
primary input. Basic legacy point Feather and raw SQLite support is also
available.

## Installation

Use your existing conda/mamba environment, then install this repository from its
root into that environment:

```bash
python -m pip install -e ".[notebooks]"
```

The reader requires Python 3.10+, NumPy, pandas and PyArrow. The notebook extra
adds Matplotlib, Cartopy, pyproj and a Jupyter kernel. Select the same environment
as the notebook kernel. For the reader alone, use `python -m pip install -e .`.

## Supported inputs

Each QC v1 period folder, `<period>/`, holds `<period>_qc.parquet`,
`<period>_qc_removed.parquet`, `<period>_qc.manifest.json` and
`<period>_qc.sqlite`. Open the kept-link Parquet or the SQLite; keep the four
files together, as the reader uses the neighbouring files.

| Input | Contents | Reader behaviour |
| --- | --- | --- |
| QC v1 kept-link Parquet (`<period>_qc.parquet`) | Kept links with QC and original trajectory IDs, endpoint state, `qc_flags`, `pair_direct_share`, SIC and neighbour counts | Validates the QC v1 link schema and the `<period>_qc.manifest.json` sidecar (status complete, zero continuity violations, known protocol id and SHA-256, cross-checked with the Arrow `limosat.qc` metadata). Reads links directly; removed links come from `<period>_qc_removed.parquet`. |
| QC v1 SQLite (`<period>_qc.sqlite`) | Every raw observation in `<period>__qc`, with `trajectory_id` and `is_last` rewritten to QC trajectories; `qc_removed_links`; `qc_metadata` | Selects the QC table and validates `qc_metadata` as above. Excludes descriptors and single-point trajectories (opt in with `include_singletons=True`), decodes POINT WKT and derives links within QC trajectories. Original trajectory IDs come from the kept and removed links by rowid. |
| Raw SQLite (`.sqlite`, `.db`) | One raw observation table | Requires an explicit CRS and is labelled as raw data; reading it does not apply QC. Use `table=` to resolve ambiguous tables. |
| Legacy point Feather (`.feather`) | `i`, `g`, `d`, `x`, `y`, `q`, or a point table with descriptive headings | Renames legacy fields; optional image IDs and correlation can be missing. An explicit CRS is required when absent. |

QC v2 outputs (materialization-manifest SQLite, v2 and earlier publisher
displacement Parquet, and monthly displacement stores) are no longer read. A catalog SQLite is not an
observation SQLite. Choose an individual period file; opening a release or
period directory is deliberately rejected because combining periods requires an
explicit ownership policy.

## Reader usage

```python
from pathlib import Path

from limosat_analysis import open_limosat

# Point at the release; copying it into this repository is unnecessary.
period = Path("data/limosat_qc_v1_20261029/202305_202407")
data = open_limosat(period / "202305_202407_qc.parquet")
links = data.read_displacements(start="2024-01-01", stop="2024-02-01")

# Observations of multi-point QC trajectories, from QC SQLite.
qc = open_limosat(period / "202305_202407_qc.sqlite")
points = qc.read_points(start="2024-01-01", stop="2024-01-04")

# The same observations, reconstructed from kept-link endpoints.
linked_points = data.read_points(start="2024-01-01", stop="2024-01-04")
```

`read_points()` returns only observations of trajectories with at least two
points, from every input. Single-point trajectories carry no displacement. In
QC v1 they are points whose every link was removed: 884,092 of 2,059,017 QC
trajectories (14% of observations) in 202305_202407. The QC SQLite keeps them;
`include_singletons=True` returns them from SQLite or Feather. Link Parquet
cannot provide them. With the default, both QC v1 files give the same
observations.

Point headings are `image_id`, `trajectory_id`, `time_utc`, `x_m`, `y_m`, `corr`,
plus available native state and provenance. `corr` is the native correlation
field, not a newly calculated quality score. Legacy `q` is interpreted as
historical correlation; this is an assumption inherited from earlier analyses.
Use `points["corr"]`, since `points.corr` is a pandas method.

Displacement output keeps the release names, including `edge_uid`,
`source_image_id`, `target_image_id`, `source_rowid`, `target_rowid`,
`start_time_utc`, `end_time_utc`, `x0_m`, `y0_m`, `x1_m`, `y1_m`, `dx_m`, `dy_m`,
`elapsed_seconds`, `distance_m` and `speed_m_per_day`. Speed is m/day; there is no
`speed_m_s`. A common `trajectory_id` column identifies the QC trajectory
(`qc_trajectory_id`). `original_trajectory_id` is the trajectory before QC
splits. Always interpret trajectory IDs within `period`; IDs from separate
periods or processing products are not assumed to identify the same trajectory.

On QC SQLite points, `original_trajectory_id` is taken from the
`original_trajectory_id` of the kept and removed links that use each
observation's rowid. Observations in no link stay missing, and all are missing
when the kept-link Parquet is not beside the SQLite. The QC ID is never
substituted. `data.metadata["original_trajectory_ids"]` records which applies.

### QC v1 fields, flags and removed links

Kept-link Parquet also returns `trajectory_link_index`, `qc_segment`, both
endpoints' `interpolated`, `corr`, `orbit_num`, `is_last`, `stopped` and
`converged_to`, `qc_flags`, `qc_flag_reasons`, `pair_direct_share`,
`qc_sic_start`, `qc_sic_end` and `qc_neighbours_30km`. `qc_flags` bits are
`FEW_NEIGHBOURS` (0) and `LOW_PAIR_DIRECT_SHARE` (1); flags never split
trajectories. Links derived from SQLite carry the endpoint state but no flags.
Endpoint `is_last` in the link Parquet is the original (pre-QC) trajectory's
flag, so a QC trajectory ending at a removed link has `target_is_last == 0`. In
the QC SQLite, `is_last` marks the end of each QC trajectory, and so do the
`source_is_last`/`target_is_last` of links derived from it. Use
`trajectory_id` and `trajectory_link_index` for QC trajectory ends in Parquet.

```python
# Every kept link is returned by default. Opt in to dropping flagged links:
strict = data.read_displacements(
    start="2024-01-01", stop="2024-02-01",
    exclude_flags=["LOW_PAIR_DIRECT_SHARE"],
)

# Links removed by QC, with the removing step and its diagnostics.
removed = data.read_removed_links(start="2024-01-01", stop="2024-02-01")
removed.qc_exclusion.value_counts()
```

`LOW_PAIR_DIRECT_SHARE` marks image pairs with less than 40% direct matches,
where kept interpolated links are more often wrong. Filter on
`pair_direct_share` directly for other thresholds. Removed links have no QC
trajectory; select them with `original_trajectory_ids=`. From SQLite they are
read from `qc_removed_links`, and from Parquet from
`<period>_qc_removed.parquet`. `qc_exclusion` holds one step code (`SPEED`,
`OPEN_WATER`, `LAND`, `OPEN_WATER+LAND`, `ISOLATED_UNVERIFIED`, `ROGUE_VECTOR`
or `INTERP_UNSUPPORTED`).

Times are normalized to timezone-aware UTC nanosecond timestamps. Naive query
dates are interpreted as UTC; all date intervals are **start inclusive, stop
exclusive**. SQLite input uses LiMOSAT's sortable, naive UTC datetime text.
Coordinates and displacements are float64 metres. Image IDs use nullable Int64
so missing legacy IDs do not require floating-point conversion. Times, IDs,
coordinates and correlation are never silently dropped because a threshold was
applied.

QC v1 data use EPSG:3413, as the protocol specifies; the release carries no CRS
in Arrow metadata. For other point files, supply `crs="EPSG:3412"` or the
correct source CRS. `crs=` declares the source projection; it does not
reproject coordinates and cannot override a conflicting known CRS.

### Full trajectories passing through a query

```python
selected, domain_points = data.read_trajectories_through(
    start="2024-02-20", stop="2024-02-25",
    xlim=(500_000, 900_000), ylim=(-1_000_000, -600_000),
)
```

This finds trajectories with at least one observation inside the inclusive
spatial box and half-open date interval, then returns their complete available
histories in the selected period. `domain_points` retains indices from
`selected`. It does not test segment intersections with the box.

Parquet endpoint reconstruction removes shared endpoint duplicates. Point
coverage is recorded in the returned DataFrame's `attrs`.

### Filtering and practical limits

```python
# Read a small set of columns and an output preview from Parquet.
preview = data.read_displacements(
    start="2024-01-01", stop="2024-02-01",
    columns=["trajectory_id", "start_time_utc", "speed_m_per_day"],
    max_rows=100,
)

# Apply the release ownership interval for non-overlapping statistics.
owned = data.read_displacements(owned_only=True)
```

Periods overlap by three months. The release ownership rule
(`limosat_release_august_annual_v1`) assigns each link to one period by its
**start time**, half-open: `201505_201607` owns 2015-05-01 to 2016-08-01, each
later period owns 1 August to 1 August, and `202405_202507` ends 2025-08-01.
The intervals match the `canonical_ownership` of the ten-period displacement
package. `owned_only` is opt-in for `read_displacements()` and
`read_removed_links()`; full trajectory queries retain overlap histories. Raw
and legacy inputs have no ownership interval.

Parquet uses column projection and date/ID/flag predicates during scanning.
Returned results are pandas DataFrames and must still fit in memory; this is not
yet a streaming analysis interface. Prefer a bounded date window or explicit
`trajectory_ids=`. `max_rows` limits a preview and is not a representative
sample. For reconstructed points, it limits the input link preview as well, so
fewer than that number of filtered points may be returned.

SQLite date queries scan the whole table because the QC table is indexed only
by trajectory. Deriving SQLite links reads complete histories for candidate
trajectories **before** applying the link-start date filter, preserving the next
endpoint beyond `stop`. A SQLite displacement preview still needs those
histories; prefer Parquet for large drift/gridding analyses. No input indexes
are created, no source files are modified, and no correlation, interpolation or
speed threshold is applied beyond QC v1 itself. Non-positive elapsed times are
retained with missing speed; velocity analyses select positive times explicitly.

Opening QC data checks the recorded manifest status, continuity and protocol,
not a new full-file checksum or SQLite integrity sweep. Reader checks and sample
comparisons are not independent scientific accuracy validation. `data.metadata`
and DataFrame `attrs` expose the period, protocol id and SHA-256, CRS, coverage,
ownership and excluded flags.

## Notebooks

Set `RELEASE_DIR` to the QC v1 release folder and `PERIOD` to a period in the
first cell, then choose UTC dates. The defaults read
`../data/limosat_qc_v1_20261029/202305_202407/202305_202407_qc.parquet`; link
or copy the release there, or point `RELEASE_DIR` at it. No data files are
distributed with this repository. Each notebook also accepts the period's QC
SQLite.

- `notebooks/read_limosat.ipynb`: point distributions, trajectories,
  consecutive drift vectors and links removed by QC.
- `notebooks/grid_limosat.ipynb`: aggregate projected vector components from
  displacement links onto a daily grid, optionally excluding flagged links.
- `notebooks/select_limosat_trajectories.ipynb`: select complete available
  trajectories through an observation-based space-time query.

These examples plot EPSG:3413 Arctic data. The reader can accept other declared
point projections, but changing the example map projection is a separate step.

## Development checks

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The small synthetic fixtures follow the QC v1 release layout. They exercise QC
splits, single-point trajectories, original IDs by rowid, flags, removed links,
manifest validation, date and ownership boundaries, shared endpoint
reconstruction, large integer IDs, malformed geometry and source immutability.
Local inputs and outputs belong in ignored `data/` and `outputs/` directories.
