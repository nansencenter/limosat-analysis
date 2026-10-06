# LiMOSAT Analysis

Shared readers and example notebooks for LiMOSAT observations and ice-drift
analysis. QC SQLite and the current displacement Parquet package are the primary
inputs. Basic legacy point Feather support is also available.

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

| Input | Contents | Reader behaviour |
| --- | --- | --- |
| QC SQLite (`.sqlite`, `.db`) | Native observations in `<period>__qc`, QC assignments and materialization manifest | Selects the QC table, checks its recorded completion status, excludes descriptors, decodes POINT WKT and preserves post-QC boundaries. |
| Working displacement Parquet (`.parquet`) | Current v2 link schema, including QC/original IDs and endpoint state | Reads links directly, preserving native fields. |
| Earlier publisher Parquet | Public link schema with `trajectory_id` and position-type fields | Reads links directly; unavailable correlation remains missing and unavailable original IDs are not invented. |
| Monthly displacement-store directory | Complete v2 manifest and monthly Parquet partitions | Reads the recorded partitions through the same interface. |
| Legacy point Feather (`.feather`) | `i`, `g`, `d`, `x`, `y`, `q`, or a point table with descriptive headings | Renames legacy fields; optional image IDs and correlation can be missing. An explicit CRS is required when absent. |

A SQLite file with one raw observation table is also readable with an explicit
CRS. It is labelled as raw data; reading it does not apply QC. Use `table=` to
resolve ambiguous observation tables. A catalog SQLite is not an observation
SQLite. Choose an individual period file or store; opening the ten-period
package directory is deliberately rejected because combining periods requires
an explicit ownership policy.

## Reader usage

```python
from limosat_analysis import open_limosat

# Point at an existing file; copying it into this repository is unnecessary.
data = open_limosat("data/201905_202007.parquet")
links = data.read_displacements(start="2020-01-01", stop="2020-02-01")

# Native observations, including singleton trajectories, from QC SQLite.
qc = open_limosat("data/201905_202007_qc.sqlite")
points = qc.read_points(start="2020-01-01", stop="2020-01-04")

# Explicitly recover linked endpoints from displacement Parquet.
linked_points = data.read_points(
    start="2020-01-01", stop="2020-01-04", linked_only=True
)
```

Point headings are `image_id`, `trajectory_id`, `time_utc`, `x_m`, `y_m`, `corr`,
plus available native state and provenance. `corr` is the native correlation
field, not a newly calculated quality score. Legacy `q` is interpreted as
historical correlation; this is an assumption inherited from earlier analyses.
Use `points["corr"]`, since `points.corr` is a pandas method.

Displacement output keeps the published names, including `source_image_id`,
`target_image_id`, `start_time_utc`, `end_time_utc`, `x0_m`, `y0_m`, `x1_m`,
`y1_m`, `dx_m`, `dy_m`, `elapsed_seconds` and `speed_m_per_day`. A common
`trajectory_id` column identifies the active trajectory; v2 also retains
`qc_trajectory_id` and `original_trajectory_id`. Always interpret trajectory IDs
within `canonical_source_window`. IDs from separate processing products or
source windows are not assumed to identify the same trajectory.

Times are normalized to timezone-aware UTC nanosecond timestamps. Naive query
dates are interpreted as UTC; all date intervals are **start inclusive, stop
exclusive**. SQLite input uses LiMOSAT's sortable, naive UTC datetime text.
Coordinates and displacements are float64 metres. Image IDs use nullable Int64
so missing legacy IDs do not require floating-point conversion. Times, IDs,
coordinates and correlation are never silently dropped because a threshold was
applied.

The Arctic QC and v2 displacement profiles use EPSG:3413. Earlier publisher
Parquet carries its CRS in Arrow metadata. The working v2 package has no embedded
CRS metadata, so the reader uses the established v2 schema's EPSG:3413 contract.
For other point files, supply `crs="EPSG:3412"` or the correct source CRS.
`crs=` declares the source projection; it does not reproject coordinates and
cannot override a conflicting known CRS.

### Full trajectories passing through a query

```python
selected, domain_points = data.read_trajectories_through(
    start="2020-02-20", stop="2020-02-25",
    xlim=(500_000, 900_000), ylim=(-1_000_000, -600_000),
    linked_only=True,
)
```

This finds trajectories with at least one observation inside the inclusive
spatial box and half-open date interval, then returns their complete available
histories in the selected source period. `domain_points` retains indices from
`selected`. It does not test segment intersections with the box.

Parquet endpoint reconstruction removes shared endpoint duplicates. It cannot
recover singleton observations, or observations absent from the link product.
`read_points()` therefore requires `linked_only=True` for displacement Parquet.
Native SQLite remains the source for complete observation counts. Point coverage
is recorded in the returned DataFrame's `attrs`.

### Filtering and practical limits

```python
# Read a small set of columns and an output preview from Parquet.
preview = data.read_displacements(
    start="2020-01-01", stop="2020-02-01",
    columns=["trajectory_id", "start_time_utc", "speed_m_per_day"],
    max_rows=100,
)

# Apply this period's manifest ownership interval for non-overlapping statistics.
owned = data.read_displacements(owned_only=True)
```

The ownership interval is applied to link **start time**. It is opt-in and
requires a recorded interval; full trajectory queries retain overlap histories.
Historical absolute paths in sidecars are not used to locate the data. The
reader follows the selected file or the store's relative partition paths.

Parquet uses column projection and date/ID predicates during scanning. Returned
results are pandas DataFrames and must still fit in memory; this is not yet a
streaming analysis interface. Prefer a bounded date window or explicit
`trajectory_ids=`. `max_rows` limits a preview and is not a representative sample.
For reconstructed points, it limits the input link preview as well, so fewer
than that number of filtered points may be returned.

SQLite date queries can scan large tables because frozen QC inputs have limited
indexes. Deriving SQLite links reads complete histories for candidate
trajectories **before** applying the link-start date filter, preserving the next
endpoint beyond `stop`. A SQLite displacement preview still needs those
histories; prefer Parquet for large drift/gridding analyses. No input indexes are
created, no source files are modified, and no QC, correlation, interpolation or
speed threshold is applied. Non-positive elapsed times are retained with missing
speed; velocity analyses select positive times explicitly.

Opening QC data checks the recorded materialization status, not a new full-file
checksum or SQLite integrity sweep. Reader checks and sample comparisons are not
independent scientific accuracy validation. `data.metadata` and DataFrame
`attrs` expose source identity, CRS, coverage and applied ownership information.

## Notebooks

Set `DATA_PATH` to an existing period Parquet or QC SQLite file, and choose UTC
dates in the first cell. The defaults use `../data/201905_202007.parquet`; no data
files are distributed with this repository.

- `notebooks/read_limosat.ipynb`: point distributions, trajectories and
  consecutive drift vectors.
- `notebooks/grid_limosat.ipynb`: aggregate projected vector components from
  displacement links onto a daily grid.
- `notebooks/select_limosat_trajectories.ipynb`: select complete available
  trajectories through an observation-based space-time query.

These examples plot EPSG:3413 Arctic data. The reader can accept other declared
point projections, but changing the example map projection is a separate step.

## Development checks

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The small fixtures exercise QC splits, singleton observations, date boundaries,
shared endpoint reconstruction, large integer IDs, missing diagnostic fields,
ownership, malformed geometry and source immutability. Local inputs and outputs
belong in ignored `data/` and `outputs/` directories.
