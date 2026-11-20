"""Reader for the public LiMOSAT trajectory files (schema 2): one Parquet point table per processing period.

A row is one observation of a feature in one Sentinel-1 scene. The rows of a trajectory share
``trajectory_id`` and are ordered by ``point_index`` (0 at the first observation). Every row after the first
also describes the displacement from the previous observation (``dx_m``, ``dy_m``, ``speed_m_per_day``,
``interpolated``, ``correlation``, ``direct_match_share``, ``quality_flags``). ``is_primary`` marks the rows in the
period's own date window; selecting those rows from all periods covers each date once.

Everything the reader needs (schema version, flag names, window, CRS, column descriptions) is read from the
file itself, so the reader carries no table of flags or periods.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .columns import (CORE_COLUMNS, CORE_DISPLACEMENT_COLUMNS, DISPLACEMENT_COLUMNS, SUPPORTED_MAJOR_VERSION, bounds as _bounds, limit as _limit, resolve_columns) 


REQUIRED_COLUMNS = ("trajectory_id", "point_index", "time", "x_m", "y_m")
NULLABLE_TYPES = {pa.bool_(): pd.BooleanDtype(), pa.uint8(): pd.UInt8Dtype()}.get


def _time_expression(field, start, stop):
    expression = ds.scalar(True)
    if start is not None:
        expression &= ds.field(field) >= pa.scalar(start, type=pa.timestamp("ns", tz="UTC"))
    if stop is not None:
        expression &= ds.field(field) < pa.scalar(stop, type=pa.timestamp("ns", tz="UTC"))
    return expression


def is_point_file(path) -> bool:
    """True if ``path`` is a Parquet file written in the public trajectory format (schema 2)."""
    path = Path(path).expanduser()
    if path.suffix != ".parquet" or not path.is_file():
        return False
    try:
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    except (OSError, pa.ArrowException):
        return False
    return b"limosat.schema_version" in metadata


def _json(raw, key, default):
    value = raw.get(key)
    return json.loads(value) if value else default


class LimosatTrajectories:
    """A public LiMOSAT trajectory file: observations of one processing period, sorted by time."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        schema = pq.ParquetFile(self.path).schema_arrow
        raw = schema.metadata or {}
        version = raw.get(b"limosat.schema_version", b"").decode()
        if not version:
            raise ValueError("Not a LiMOSAT trajectory file: no limosat.schema_version in the Parquet metadata")
        if int(version.split(".")[0]) != SUPPORTED_MAJOR_VERSION:
            raise ValueError(f"Unsupported LiMOSAT schema {version}; this reader supports {SUPPORTED_MAJOR_VERSION}.x")
        missing = set(REQUIRED_COLUMNS) - set(schema.names)
        if missing:
            raise ValueError(f"LiMOSAT trajectory file lacks columns {sorted(missing)}")
        self._schema = schema
        self._dataset = ds.dataset(self.path, format="parquet")
        self.kind = "parquet_points"
        self.metadata = {
            "schema_version": version,
            "release_id": raw.get(b"limosat.release_id", b"").decode() or None,
            "period": raw.get(b"limosat.period", b"").decode() or None,
            "crs": raw.get(b"limosat.crs", b"").decode() or None,
            "flag_bits": _json(raw, b"limosat.flag_bits", {}),
            "primary_window": _json(raw, b"limosat.primary_window", None),
            "primary_policy": raw.get(b"limosat.primary_policy", b"").decode() or None,
            "rows": pq.ParquetFile(self.path).metadata.num_rows,
        }

    def __repr__(self):
        return f"LimosatTrajectories({self.path.name!r}, period={self.metadata['period']!r}, rows={self.metadata['rows']:,})"

    # ----------------------------------------------------------------------------------------- helpers

    def describe(self) -> pd.DataFrame:
        """Column name, type, unit and description, as recorded in the file."""
        rows = []
        for field in self._schema:
            meta = field.metadata or {}
            rows.append({"column": field.name, "type": str(field.type), "unit": meta.get(b"unit", b"").decode(),
                         "description": meta.get(b"description", b"").decode()})
        return pd.DataFrame(rows)

    def _check_columns(self, columns):
        if columns is not None:
            missing = set(columns) - set(self._schema.names)
            if missing:
                raise ValueError(f"Unavailable columns: {sorted(missing)}")

    def _frame(self, table) -> pd.DataFrame:
        frame = table.to_pandas(types_mapper=NULLABLE_TYPES)
        if "time" in frame:
            frame["time"] = frame["time"].astype("datetime64[ns, UTC]")
        return frame

    def _filter(self, start, stop, trajectory_ids, primary_only):
        expression = _time_expression("time", start, stop)
        if trajectory_ids is not None:
            expression &= ds.field("trajectory_id").isin([int(i) for i in np.atleast_1d(trajectory_ids)])
        if primary_only:
            if "is_primary" not in self._schema.names:
                raise ValueError("This file has no is_primary column")
            expression &= ds.field("is_primary")
        return expression

    def _attrs(self, frame):
        frame.attrs.update(self.metadata)
        return frame

    # ----------------------------------------------------------------------------------------- readers

    def read_points(self, *, start=None, stop=None, trajectory_ids=None, columns=None, primary_only=False,
                    max_rows=None) -> pd.DataFrame:
        """Observations with ``start <= time < stop`` (UTC; naive dates are UTC), sorted by time.

        ``primary_only`` keeps rows in the period's own window. ``trajectory_ids`` are unique within this
        period only.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        columns = resolve_columns(columns, self._schema.names, CORE_COLUMNS)
        scan = list(dict.fromkeys([*columns, "time", "trajectory_id", "point_index"]))
        scanner = self._dataset.scanner(columns=scan, filter=self._filter(start, stop, trajectory_ids, primary_only))
        frame = self._frame(scanner.to_table())
        frame = frame.sort_values(["time", "trajectory_id", "point_index"], kind="stable")
        if max_rows is not None:
            frame = frame.iloc[:max_rows]
        return self._attrs(frame[columns].reset_index(drop=True))

    def _trajectory_ids_with_rows(self, start, stop, xlim, ylim):
        columns = ["trajectory_id"] + (["x_m", "y_m"] if xlim or ylim else [])
        table = self._dataset.to_table(columns=columns, filter=self._filter(start, stop, None, False))
        frame = table.to_pandas()
        if xlim is not None:
            frame = frame[(frame.x_m >= xlim[0]) & (frame.x_m <= xlim[1])]
        if ylim is not None:
            frame = frame[(frame.y_m >= ylim[0]) & (frame.y_m <= ylim[1])]
        return np.unique(frame["trajectory_id"].to_numpy())

    def read_trajectories(self, *, start=None, stop=None, xlim=None, ylim=None, trajectory_ids=None, columns=None,
                          max_rows=None) -> pd.DataFrame:
        """Complete trajectories: every observation of each trajectory that has an observation in the window.

        ``start``/``stop`` select by observation time and ``xlim``/``ylim`` (EPSG:3413 metres) by position;
        an observation inside both makes the whole trajectory part of the result. The result is ordered by
        ``trajectory_id`` and ``point_index``, so ``groupby('trajectory_id')`` yields one track at a time.
        Give ``trajectory_ids`` alone to read specific trajectories.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        columns = resolve_columns(columns, self._schema.names, CORE_COLUMNS)
        if start is not None or stop is not None or xlim is not None or ylim is not None:
            ids = self._trajectory_ids_with_rows(start, stop, xlim, ylim)
            if trajectory_ids is not None:
                ids = np.intersect1d(ids, np.atleast_1d(trajectory_ids))
        elif trajectory_ids is not None:
            ids = np.atleast_1d(trajectory_ids)
        else:
            raise ValueError("Give start/stop, xlim/ylim or trajectory_ids; to read everything use read_points()")
        wanted = list(dict.fromkeys(["trajectory_id", "point_index", *columns]))
        table = self._dataset.to_table(columns=wanted, filter=self._filter(None, None, ids, False))
        frame = self._frame(table).sort_values(["trajectory_id", "point_index"], kind="stable").reset_index(drop=True)
        frame = frame[columns]
        return self._attrs(frame.iloc[:max_rows] if max_rows is not None else frame)

    def read_displacements(self, *, start=None, stop=None, trajectory_ids=None, columns=None, primary_only=False,
                           exclude_flags=(), max_rows=None) -> pd.DataFrame:
        """One row per displacement between consecutive observations of a trajectory.

        ``start <= start_time < stop`` selects by the time the displacement starts, which is also what
        ``is_primary`` is defined on. Columns: ``trajectory_id``, ``point_index`` (of the end observation),
        ``start_time``, ``end_time``, ``elapsed_seconds`` (derived from the two times), ``x0_m``, ``y0_m``, ``x1_m``,
        ``y1_m``, the displacement columns,
        ``start_scene``/``end_scene``, ``is_primary`` and ``period`` where the file has them.
        ``columns`` is None for the core columns, ``"all"`` or a list of names. ``exclude_flags`` names flags (see ``metadata['flag_bits']``) whose displacements are dropped; by
        default every displacement is returned.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        if isinstance(exclude_flags, str):
            exclude_flags = (exclude_flags,)
        bits = self.metadata["flag_bits"]
        unknown = set(exclude_flags) - set(bits)
        if unknown:
            raise ValueError(f"Unknown flags: {sorted(unknown)}; this file declares {sorted(bits)}")
        if start is not None or stop is not None:
            ids = self._trajectory_ids_with_rows(start, stop, None, None)
            if trajectory_ids is not None:
                ids = np.intersect1d(ids, np.atleast_1d(trajectory_ids))
        else:
            ids = trajectory_ids
        rows = self._frame(self._dataset.to_table(filter=self._filter(None, None, ids, False)))
        rows = rows.sort_values(["trajectory_id", "point_index"], kind="stable").reset_index(drop=True)
        previous = rows.groupby("trajectory_id", sort=False).shift(1)
        has_previous = previous["time"].notna().to_numpy()
        end, before = rows[has_previous], previous[has_previous]
        frame = pd.DataFrame({"trajectory_id": end["trajectory_id"], "point_index": end["point_index"],
                              "start_time": before["time"], "end_time": end["time"],
                              "elapsed_seconds": (end["time"] - before["time"]).dt.total_seconds(),
                              "x0_m": before["x_m"], "y0_m": before["y_m"], "x1_m": end["x_m"], "y1_m": end["y_m"]})
        for name in DISPLACEMENT_COLUMNS:
            if name in end:
                frame[name] = end[name]
        if "scene" in end:
            frame["start_scene"], frame["end_scene"] = before["scene"], end["scene"]
        for name in ("is_primary", "period"):
            if name in end:
                frame[name] = end[name]
        keep = pd.Series(True, index=frame.index)
        if start is not None:
            keep &= frame["start_time"] >= start
        if stop is not None:
            keep &= frame["start_time"] < stop
        if primary_only:
            if "is_primary" not in frame:
                raise ValueError("This file has no is_primary column")
            keep &= frame["is_primary"].fillna(False).astype(bool)
        if exclude_flags:
            mask = sum(1 << bits[name] for name in set(exclude_flags))
            keep &= (frame["quality_flags"].fillna(0).astype("int64") & mask) == 0
        frame = frame[keep].sort_values(["start_time", "trajectory_id", "point_index"], kind="stable")
        frame.attrs["excluded_flags"] = sorted(set(exclude_flags))
        if max_rows is not None:
            frame = frame.iloc[:max_rows]
        columns = resolve_columns(columns, list(frame.columns), CORE_DISPLACEMENT_COLUMNS)
        return self._attrs(frame[columns].reset_index(drop=True))

    def verify(self) -> bool:
        """Check this file's SHA-256 against ``manifest.json`` in the same folder (raises on mismatch)."""
        manifest_path = self.path.with_name("manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No manifest.json beside {self.path.name}")
        entries = [e for e in json.loads(manifest_path.read_text())["files"] if e["file"] == self.path.name]
        if len(entries) != 1:
            raise ValueError(f"{self.path.name} is not listed once in {manifest_path.name}")
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != entries[0]["sha256"]:
            raise ValueError(f"{self.path.name}: checksum differs from manifest.json")
        return True


def open_limosat(path):
    """Open one period file of the public LiMOSAT trajectory release.

    ``limosat_<period>.nc`` (NetCDF4, the released format) and ``limosat_<period>.parquet`` (the same table as Parquet) are
    both read through the same methods: ``read_points``, ``read_trajectories``, ``read_displacements``, ``describe``,
    ``verify``. Reads return the core columns unless ``columns`` is given (a list of names, or ``"all"``).
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise ValueError("Open one period file, limosat_<period>.nc; periods are not combined automatically")
    if path.suffix == ".nc":
        from .netcdf import LimosatNetCDF, is_netcdf_file

        if not is_netcdf_file(path):
            raise ValueError(f"{path.name} is not a LiMOSAT trajectory file (NetCDF4 with schema_version and featureType "
                             "trajectory attributes)")
        return LimosatNetCDF(path)
    if not is_point_file(path):
        raise ValueError(
            f"{path.name} is not a LiMOSAT trajectory file (NetCDF4, or Parquet with limosat.schema_version metadata). "
            "Raw tracking databases, the earlier benchmark Feather files and the internal QC v1 archive are not read "
            "by this version; use limosat-analysis commit 6d3694b for them (pandas.read_feather reads the Feather files)."
        )
    return LimosatTrajectories(path)
