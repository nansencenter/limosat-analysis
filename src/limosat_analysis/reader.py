"""Read-only adapters for LiMOSAT QC SQLite and displacement Parquet."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as feather
import pyarrow.parquet as pq

POINT_COLUMNS = ["image_id", "trajectory_id", "time_utc", "x_m", "y_m", "corr"]
LINK_COLUMNS = [
    "source_image_id", "target_image_id", "start_time_utc", "end_time_utc",
    "x0_m", "y0_m", "x1_m", "y1_m", "elapsed_seconds", "dx_m", "dy_m",
    "distance_m", "speed_m_s", "speed_m_per_day",
]
POINT_PATTERN = r"^\s*POINT\s*(?:Z\s*)?\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)(?:\s+[-+0-9.eE]+)?\s*\)\s*$"


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _connect(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _utc(value):
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Time bounds must not be NaT")
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _bounds(start, stop):
    start, stop = _utc(start), _utc(stop)
    if start is not None and stop is not None and start >= stop:
        raise ValueError("Expected start < stop (stop is exclusive)")
    return start, stop


def _time_mask(values, start, stop):
    mask = pd.Series(True, index=values.index)
    if start is not None:
        mask &= values >= start
    if stop is not None:
        mask &= values < stop
    return mask


def _finish(frame, columns, metadata, max_rows=None):
    keys = [key for key in ("canonical_source_window", "trajectory_id", "time_utc", "start_time_utc", "image_id", "source_image_id", "point_rowid", "source_rowid") if key in frame]
    if keys:
        frame = frame.sort_values(keys, kind="stable")
    if max_rows is not None:
        frame = frame.iloc[:max_rows]
    if columns is None and "time_utc" in frame:
        first = [column for column in POINT_COLUMNS if column in frame]
        frame = frame[first + [column for column in frame if column not in first]]
    if columns is not None:
        missing = set(columns) - set(frame)
        if missing:
            raise ValueError(f"Unavailable columns: {sorted(missing)}")
        frame = frame.loc[:, list(columns)]
    frame = frame.reset_index(drop=True)
    frame.attrs.update(metadata)
    return frame


def _limit(max_rows):
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1):
        raise ValueError("max_rows must be a positive integer")


class LimosatDataset:
    """A single source period, with explicit point and displacement views.

    Dates are half-open UTC intervals. ``max_rows`` is a preview limit, not a
    statistically representative sample. Connections never modify input files.
    """

    def __init__(self, path, *, table=None, crs=None):
        self.path = Path(path).expanduser().resolve()
        self.table = None
        self._manifest = {}
        self._arrow = None
        self._original_ids = None
        self.metadata = {"source_path": str(self.path), "crs": None, "qc_materialized": False}
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.suffix.lower() in {".sqlite", ".db"}:
            self._inspect_sqlite(table)
        elif self.path.is_dir() or self.path.suffix.lower() == ".parquet":
            if table is not None:
                raise ValueError("table applies only to SQLite")
            self._inspect_parquet()
        elif self.path.suffix.lower() == ".feather":
            if table is not None:
                raise ValueError("table applies only to SQLite")
            self.kind = "feather_points"
            self.metadata["legacy_quality_assumption"] = "q is interpreted as historical correlation"
        else:
            raise ValueError("Expected a SQLite, Parquet, Feather file or a monthly displacement-store directory")
        known_crs = self.metadata["crs"]
        if crs is not None and known_crs is not None and crs.upper() != known_crs.upper():
            raise ValueError(f"Source CRS is {known_crs}; crs does not reproject coordinates")
        self.metadata["crs"] = known_crs or crs
        if self.metadata["crs"] is None:
            raise ValueError("Source has no CRS metadata; supply crs explicitly")
        self.metadata.setdefault("canonical_source_window", self.path.stem)
        self.metadata["kind"] = self.kind
        self.metadata["time_convention"] = "UTC; start inclusive, stop exclusive"
        self.metadata["coordinate_units"] = "metres"

    def _inspect_sqlite(self, table):
        with _connect(self.path) as conn:
            names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            candidates = []
            for name in names:
                fields = {row[1] for row in conn.execute(f"PRAGMA table_info({_quote(name)})")}
                if {"image_id", "trajectory_id", "time", "geometry"} <= fields:
                    candidates.append(name)
            if table is None:
                cleaned = [name for name in candidates if name.endswith("__qc")]
                choices = cleaned or candidates
                if len(choices) != 1:
                    raise ValueError(f"Choose table explicitly; observation tables: {choices}")
                table = choices[0]
            if table not in candidates:
                raise ValueError(f"Not a LiMOSAT observation table: {table!r}")
            self.table = table
            self._point_fields = [row[1] for row in conn.execute(f"PRAGMA table_info({_quote(table)})") if row[1] != "descriptors"]
            self.kind = "sqlite_points"
            samples = conn.execute(f"SELECT time FROM {_quote(table)} WHERE time IS NOT NULL LIMIT 3").fetchall()
            for (value,) in samples:
                if not isinstance(value, str) or len(value) < 19 or value[10] != " " or pd.Timestamp(value).tzinfo is not None:
                    raise ValueError("Expected LiMOSAT SQLite times as naive UTC 'YYYY-MM-DD HH:MM:SS' text")
            if table.endswith("__qc"):
                if "qc_materialization_manifest" not in names:
                    raise ValueError("QC table has no materialization manifest")
                row = conn.execute("SELECT manifest_json FROM qc_materialization_manifest LIMIT 1").fetchone()
                manifest = json.loads(row[0]) if row else {}
                if manifest.get("status") != "complete" or manifest.get("quick_check") != "ok":
                    raise ValueError("QC materialization is not recorded as complete with quick_check: ok")
                if manifest.get("cleaned_table", table) != table:
                    raise ValueError("Selected table differs from the QC materialization manifest")
                self.metadata.update(qc_materialized=True, crs="EPSG:3413")
                self.metadata["canonical_source_window"] = table.removesuffix("__qc")
            else:
                self.metadata["canonical_source_window"] = table

    def _inspect_parquet(self):
        if self.path.is_dir():
            manifest_path = self.path / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError("Monthly displacement store requires manifest.json")
            self._manifest = json.loads(manifest_path.read_text())
            if "windows" in self._manifest:
                raise ValueError("Choose one period Parquet file; multi-period concatenation needs explicit ownership")
            if self._manifest.get("status") != "complete" or self._manifest.get("store_format_version") != 2:
                raise ValueError("Expected a complete v2 monthly displacement store")
            files = []
            for record in self._manifest["partitions"]:
                part = (self.path / record["path"]).resolve()
                if not part.is_relative_to(self.path):
                    raise ValueError("Partition path escapes the displacement store")
                files.append(str(part))
        else:
            files = [str(self.path)]
            sidecar = self.path.with_suffix(".manifest.json")
            if sidecar.is_file():
                self._manifest = json.loads(sidecar.read_text())
            else:
                for parent in list(self.path.parents)[:5]:
                    manifest_path = parent / "manifest.json"
                    if manifest_path.is_file():
                        candidate = json.loads(manifest_path.read_text())
                        if candidate.get("store_format_version") == 2:
                            self._manifest = candidate
                            break
        if not files:
            raise ValueError("No Parquet partitions recorded in store")
        schema = pq.ParquetFile(files[0]).schema_arrow
        for file in files[1:]:
            if not pq.ParquetFile(file).schema_arrow.equals(schema):
                raise ValueError(f"Partition schema differs: {file}")
        fields = set(schema.names)
        for name in ("start_time_utc", "end_time_utc"):
            if name in fields:
                dtype = schema.field(name).type
                if not pa.types.is_timestamp(dtype) or dtype.tz != "UTC":
                    raise ValueError("Expected UTC timestamp columns in displacement Parquet")
        if not set(LINK_COLUMNS) <= fields or not ({"trajectory_id", "qc_trajectory_id"} & fields):
            raise ValueError("Expected the LiMOSAT displacement Parquet schema")
        if "qc_trajectory_id" in fields:
            if not {"canonical_source_window", "source_rowid", "target_rowid", "original_trajectory_id"} <= fields:
                raise ValueError("Incomplete v2 displacement identity columns")
            self.kind = "parquet_displacements_v2"
            self.metadata.update(crs="EPSG:3413", qc_materialized=True)
        else:
            self.kind = "parquet_displacements_v1"
        arrow_metadata = {key.decode(): value.decode() for key, value in (schema.metadata or {}).items() if key.startswith(b"limosat.")}
        self.metadata["crs"] = arrow_metadata.get("limosat.crs", self.metadata["crs"])
        window = self._manifest.get("window") or arrow_metadata.get("limosat.source_window")
        if window:
            self.metadata["canonical_source_window"] = window
        release = self._manifest.get("source_qc_release_id") or self._manifest.get("qc_release_id") or arrow_metadata.get("limosat.release_id")
        if release:
            self.metadata["qc_release_id"] = release
        if self.kind == "parquet_displacements_v1" and release:
            self.metadata["qc_materialized"] = True
        self.metadata["canonical_ownership"] = self._manifest.get("canonical_ownership")
        self._arrow = ds.dataset(files, format="parquet")
        self._link_fields = fields

    def _original_map(self):
        if self._original_ids is None:
            mapping = {}
            with _connect(self.path) as conn:
                names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table in ("qc_break_assignment", "qc_duplicate_assignment"):
                    if table in names:
                        rows = conn.execute(f"SELECT new_trajectory_id, original_trajectory_id FROM {table} WHERE new_trajectory_id IS NOT NULL")
                        for new, original in rows:
                            if new in mapping and mapping[new] != original:
                                raise ValueError("QC trajectory maps to conflicting original IDs")
                            mapping[new] = original
            self._original_ids = mapping
        return self._original_ids

    def _normalize_points(self, frame):
        if "time" in frame:
            frame = frame.rename(columns={"time": "time_utc"})
        frame["time_utc"] = pd.to_datetime(frame.time_utc, utc=True, errors="raise").astype("datetime64[ns, UTC]")
        if frame.time_utc.isna().any():
            raise ValueError("Missing observation timestamp")
        if "geometry" in frame:
            xy = frame.pop("geometry").astype("string").str.extract(POINT_PATTERN)
            if xy.isna().any().any():
                raise ValueError("Malformed or missing POINT geometry in LiMOSAT observations")
            frame[["x_m", "y_m"]] = xy.astype("float64").to_numpy()
        if not np.isfinite(frame[["x_m", "y_m"]].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite point coordinates")
        frame["trajectory_id"] = frame.trajectory_id.astype("int64")
        for field in ("image_id", "point_rowid", "orbit_num", "converged_to"):
            if field in frame:
                frame[field] = frame[field].astype("Int64")
        if "image_id" not in frame:
            frame["image_id"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        if "corr" not in frame:
            frame["corr"] = np.nan
        if self.metadata.get("qc_materialized"):
            frame["qc_trajectory_id"] = frame.trajectory_id
            mapping = self._original_map() if self.kind == "sqlite_points" else {}
            if self.kind == "sqlite_points":
                frame["original_trajectory_id"] = pd.array([mapping.get(value, value) for value in frame.trajectory_id], dtype="Int64")
        frame["canonical_source_window"] = self.metadata.get("canonical_source_window", self.path.stem)
        return frame

    def _point_query(self, start=None, stop=None, trajectory_ids=None):
        where, params = [], []
        # LiMOSAT SQLite stores UTC as sortable naive datetime text. Text
        # comparisons preserve microseconds that julianday would round away.
        if start is not None:
            where.append("time >= ?")
            params.append(start.tz_localize(None).isoformat(sep=" "))
        if stop is not None:
            where.append("time < ?")
            params.append(stop.tz_localize(None).isoformat(sep=" "))
        if trajectory_ids is not None:
            where.append("trajectory_id IN (SELECT CAST(value AS INTEGER) FROM json_each(?))")
            params.append(json.dumps([int(value) for value in trajectory_ids]))
        suffix = " WHERE " + " AND ".join(where) if where else ""
        return suffix, params

    def read_points(self, *, start=None, stop=None, trajectory_ids=None, columns=None, linked_only=False, max_rows=None):
        """Read observations, or explicitly reconstruct linked Parquet endpoints.

        Link Parquet cannot recover singleton trajectories. Missing image IDs or
        correlation values in legacy Feather remain missing, never fabricated.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        if self._arrow is not None:
            if not linked_only:
                raise ValueError("Displacement Parquet excludes singleton observations; use linked_only=True or the QC SQLite for complete points")
            links = self._read_parquet(start, stop, trajectory_ids, endpoint_times=True, max_rows=max_rows)
            frame = self._endpoints(links)
            frame = frame.loc[_time_mask(frame.time_utc, start, stop)].copy()
            metadata = {**self.metadata, "point_coverage": "linked endpoints only; singletons unavailable"}
        elif self.kind == "sqlite_points":
            suffix, params = self._point_query(start, stop, trajectory_ids)
            query = f"SELECT rowid AS point_rowid, {', '.join(map(_quote, self._point_fields))} FROM {_quote(self.table)}{suffix}"
            if max_rows is not None:
                query += " LIMIT ?"
                params.append(max_rows)
            with _connect(self.path) as conn:
                frame = pd.read_sql_query(query, conn, params=params)
            frame = self._normalize_points(frame)
            metadata = {**self.metadata, "point_coverage": "native observations"}
        else:
            frame = feather.read_table(self.path).to_pandas().rename(columns={"i": "image_id", "g": "trajectory_id", "d": "time_utc", "x": "x_m", "y": "y_m", "q": "corr"})
            missing = {"trajectory_id", "time_utc", "x_m", "y_m"} - set(frame)
            if missing:
                raise ValueError(f"Feather point table missing columns: {sorted(missing)}")
            frame = self._normalize_points(frame)
            mask = _time_mask(frame.time_utc, start, stop)
            if trajectory_ids is not None:
                mask &= frame.trajectory_id.isin(trajectory_ids)
            frame = frame.loc[mask].copy()
            metadata = {**self.metadata, "point_coverage": "native observations"}
        return _finish(frame, columns, metadata, max_rows)

    def _read_parquet(self, start, stop, trajectory_ids, *, endpoint_times=False, columns=None, max_rows=None):
        def time_filter(field):
            expression = ds.scalar(True)
            for value, comparison in ((start, "start"), (stop, "stop")):
                if value is not None:
                    scalar = pa.scalar(value, type=pa.timestamp("ns", tz="UTC"))
                    expression &= ds.field(field) >= scalar if comparison == "start" else ds.field(field) < scalar
            return expression
        expression = time_filter("start_time_utc")
        if endpoint_times:
            expression |= time_filter("end_time_utc")
        id_field = "qc_trajectory_id" if "qc_trajectory_id" in self._link_fields else "trajectory_id"
        if trajectory_ids is not None:
            expression &= ds.field(id_field).isin([int(value) for value in trajectory_ids])
        scan_columns = None
        if columns is not None:
            virtual = {"trajectory_id", "source_corr", "target_corr", "canonical_source_window", "qc_release_id", "source_interpolated", "target_interpolated"}
            missing = set(columns) - self._link_fields - virtual
            if missing:
                raise ValueError(f"Unavailable columns: {sorted(missing)}")
            needed = set(columns) & self._link_fields
            if "trajectory_id" in columns:
                needed.add(id_field)
            for prefix in ("source", "target"):
                if f"{prefix}_interpolated" in columns and f"{prefix}_position_type" in self._link_fields:
                    needed.add(f"{prefix}_position_type")
            # Keep ordering independent of output projection.
            needed |= {id_field, "start_time_utc", "source_image_id"}
            needed |= self._link_fields & {"canonical_source_window", "source_rowid"}
            scan_columns = sorted(needed)
        scanner = self._arrow.scanner(filter=expression, columns=scan_columns, batch_size=100_000)
        if max_rows is None:
            frame = scanner.to_table().to_pandas()
        else:
            frame = scanner.head(max_rows).to_pandas()
        if id_field in frame:
            frame["trajectory_id"] = frame[id_field].astype("int64")
        for name in ("source_image_id", "target_image_id"):
            if name in frame:
                frame[name] = frame[name].astype("Int64")
        for prefix in ("source", "target"):
            corr = f"{prefix}_corr"
            if corr not in frame:
                frame[corr] = np.nan
            interp, position = f"{prefix}_interpolated", f"{prefix}_position_type"
            if interp not in frame and position in frame:
                if not frame[position].isin(["observed", "interpolated"]).all():
                    raise ValueError(f"Unknown {position}")
                frame[interp] = frame[position].eq("interpolated").astype("int8")
        for field in ("canonical_source_window", "qc_release_id"):
            if field not in frame and field in self.metadata:
                frame[field] = self.metadata[field]
        if "canonical_source_window" in frame and frame.canonical_source_window.nunique(dropna=False) > 1:
            raise ValueError("Expected one source window per dataset; select periods independently")
        for field in ("start_time_utc", "end_time_utc"):
            if field in frame:
                frame[field] = pd.to_datetime(frame[field], utc=True, errors="raise").astype("datetime64[ns, UTC]")
        return frame

    def _endpoints(self, links):
        frames = []
        for prefix, endpoint, time in (("source", "0", "start_time_utc"), ("target", "1", "end_time_utc")):
            mapping = {f"{prefix}_image_id": "image_id", time: "time_utc", f"x{endpoint}_m": "x_m", f"y{endpoint}_m": "y_m"}
            for name in ("corr", "interpolated", "orbit_num", "is_last", "stopped", "converged_to"):
                mapping[f"{prefix}_{name}"] = name
            mapping[f"{prefix}_rowid"] = "point_rowid"
            shared = [field for field in ("trajectory_id", "qc_trajectory_id", "original_trajectory_id", "canonical_source_window", "qc_release_id") if field in links]
            fields = [field for field in mapping if field in links]
            frames.append(links[shared + fields].rename(columns=mapping))
        frame = pd.concat(frames, ignore_index=True)
        for name in ("image_id", "point_rowid"):
            if name in frame:
                frame[name] = frame[name].astype("Int64")
        keys = [key for key in ("canonical_source_window", "trajectory_id") if key in frame]
        keys += ["point_rowid"] if "point_rowid" in frame else ["image_id", "time_utc"]
        # An internal point occurs once as a target and once as a source.
        state = [field for field in ("image_id", "time_utc", "x_m", "y_m", "corr", "interpolated", "is_last", "orbit_num", "stopped", "converged_to") if field in frame and field not in keys]
        if frame.drop_duplicates(keys + state).duplicated(keys).any():
            raise ValueError("Displacement endpoints disagree about a shared observation")
        return frame.drop_duplicates(keys).copy()

    def read_displacements(self, *, start=None, stop=None, trajectory_ids=None, columns=None, owned_only=False, max_rows=None):
        """Read consecutive links, filtered by their start time after pairing.

        QC boundaries are preserved. No correlation, speed or interpolation
        threshold is applied. Non-positive elapsed times yield missing speed.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        if owned_only:
            ownership = self.metadata.get("canonical_ownership")
            if not ownership:
                raise ValueError("Source has no recorded canonical ownership interval")
            owner_start, owner_stop = _bounds(ownership["start"], ownership["stop_exclusive"])
            start = max(start, owner_start) if start is not None else owner_start
            stop = min(stop, owner_stop) if stop is not None else owner_stop
            if start >= stop:
                # Produce an empty, typed result through the normal read path.
                start = stop
        if self._arrow is not None:
            frame = self._read_parquet(start, stop, trajectory_ids, columns=columns, max_rows=max_rows)
        else:
            ids = trajectory_ids
            if self.kind == "sqlite_points" and ids is None and (start is not None or stop is not None):
                suffix, params = self._point_query(start, stop)
                with _connect(self.path) as conn:
                    ids = [row[0] for row in conn.execute(f"SELECT DISTINCT trajectory_id FROM {_quote(self.table)}{suffix}", params)]
            # Fetch complete histories before pairing: filtering points first
            # would remove links whose target lies beyond the stop boundary.
            points = self.read_points(trajectory_ids=ids)
            frame = self._pair_points(points)
            frame = frame.loc[_time_mask(frame.start_time_utc, start, stop)].copy()
        metadata = {**self.metadata, "ownership_applied": owned_only, "time_filter": "displacement start time"}
        return _finish(frame, columns, metadata, max_rows)

    def _pair_points(self, points):
        keys = ["canonical_source_window", "trajectory_id"]
        following = points.groupby(keys, sort=False).shift(-1)
        keep = following.time_utc.notna()
        if "is_last" in points:
            keep &= points.is_last.eq(0)
        source, target = points.loc[keep], following.loc[keep]
        shared = [field for field in ("canonical_source_window", "trajectory_id", "qc_trajectory_id", "original_trajectory_id") if field in points]
        frame = source[shared].copy()
        for prefix, endpoint, time in (("source", "0", "start_time_utc"), ("target", "1", "end_time_utc")):
            values = source if prefix == "source" else target
            frame[f"{prefix}_image_id"] = values.image_id.astype("Int64")
            frame[time] = values.time_utc
            frame[f"x{endpoint}_m"], frame[f"y{endpoint}_m"] = values.x_m, values.y_m
            for field in ("corr", "interpolated", "orbit_num", "is_last", "stopped", "converged_to"):
                if field in values:
                    frame[f"{prefix}_{field}"] = values[field]
            if "point_rowid" in values:
                frame[f"{prefix}_rowid"] = values.point_rowid.astype("int64")
        frame["elapsed_seconds"] = (frame.end_time_utc - frame.start_time_utc).dt.total_seconds()
        frame["positive_elapsed"] = frame.elapsed_seconds > 0
        frame["dx_m"] = frame.x1_m - frame.x0_m
        frame["dy_m"] = frame.y1_m - frame.y0_m
        frame["distance_m"] = np.hypot(frame.dx_m, frame.dy_m)
        frame["speed_m_s"] = frame.distance_m / frame.elapsed_seconds.where(frame.positive_elapsed)
        frame["speed_m_per_day"] = frame.speed_m_s * 86_400
        return frame

    def read_trajectories_through(self, *, start, stop, xlim, ylim, linked_only=False):
        """Return full trajectories and matching points for an endpoint query.

        Spatial bounds are inclusive projected metres. A trajectory matches if
        at least one observation lies inside the box and half-open time window;
        this is not a geometric segment-intersection query.
        """
        start, stop = _bounds(start, stop)
        if xlim[0] > xlim[1] or ylim[0] > ylim[1]:
            raise ValueError("Expected increasing xlim and ylim bounds")
        candidates = self.read_points(start=start, stop=stop, linked_only=linked_only)
        inside = candidates.x_m.between(*xlim) & candidates.y_m.between(*ylim)
        ids = candidates.loc[inside, "trajectory_id"].unique()
        selected = self.read_points(trajectory_ids=ids, linked_only=linked_only)
        in_domain = _time_mask(selected.time_utc, start, stop) & selected.x_m.between(*xlim) & selected.y_m.between(*ylim)
        domain = selected.loc[in_domain].copy()
        return selected, domain


def open_limosat(path, *, table=None, crs=None):
    """Inspect a LiMOSAT source without loading its observation/link table."""
    return LimosatDataset(path, table=table, crs=crs)
