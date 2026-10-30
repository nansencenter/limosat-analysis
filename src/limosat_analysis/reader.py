"""Read-only adapters for LiMOSAT QC v1 releases and legacy point files."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as feather
import pyarrow.parquet as pq

POINT_COLUMNS = ["image_id", "trajectory_id", "time_utc", "x_m", "y_m", "corr"]
LINK_COLUMNS = [
    "source_image_id", "target_image_id", "start_time_utc", "end_time_utc",
    "x0_m", "y0_m", "x1_m", "y1_m", "elapsed_seconds", "dx_m", "dy_m",
    "distance_m", "speed_m_per_day",
]
ENDPOINT_FIELDS = ("interpolated", "corr", "orbit_num", "is_last", "stopped", "converged_to")
QC_BASE_COLUMNS = [
    "edge_uid", "original_trajectory_id", "trajectory_link_index", "source_rowid", "target_rowid",
    *LINK_COLUMNS, *(f"{prefix}_{name}" for prefix in ("source", "target") for name in ENDPOINT_FIELDS),
    "pair_direct_share",
]
QC_LINK_COLUMNS = QC_BASE_COLUMNS + [
    "qc_trajectory_id", "qc_segment", "qc_flags", "qc_flag_reasons",
    "qc_sic_start", "qc_sic_end", "qc_neighbours_30km",
]
QC_REMOVED_COLUMNS = QC_BASE_COLUMNS + ["qc_exclusion"]
QC_PROTOCOLS = {"limosat_qc_v1_draft_20261028": "e91d7a9c8e9588650f90125ea804d4d04aae629dca422aceec6efb878394fff3"}
QC_FLAG_BITS = {"FEW_NEIGHBOURS": 0, "LOW_PAIR_DIRECT_SHARE": 1}
QC_CRS = "EPSG:3413"
# Release date ownership of link start times, half-open. Matches the
# canonical_ownership intervals of the ten-period displacement package.
OWNERSHIP_POLICY = "limosat_release_august_annual_v1"
OWNERSHIP = {
    "201505_201607": ("2015-05-01", "2016-08-01"),
    "201605_201707": ("2016-08-01", "2017-08-01"),
    "201705_201807": ("2017-08-01", "2018-08-01"),
    "201805_201907": ("2018-08-01", "2019-08-01"),
    "201905_202007": ("2019-08-01", "2020-08-01"),
    "202005_202107": ("2020-08-01", "2021-08-01"),
    "202105_202207": ("2021-08-01", "2022-08-01"),
    "202205_202307": ("2022-08-01", "2023-08-01"),
    "202305_202407": ("2023-08-01", "2024-08-01"),
    "202405_202507": ("2024-08-01", "2025-08-01"),
}
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


def _time_expression(field, start, stop):
    expression = ds.scalar(True)
    if start is not None:
        expression &= ds.field(field) >= pa.scalar(start, type=pa.timestamp("ns", tz="UTC"))
    if stop is not None:
        expression &= ds.field(field) < pa.scalar(stop, type=pa.timestamp("ns", tz="UTC"))
    return expression


def _finish(frame, columns, metadata, max_rows=None):
    keys = [key for key in ("period", "trajectory_id", "original_trajectory_id", "time_utc", "trajectory_link_index", "start_time_utc", "image_id", "source_image_id", "point_rowid", "source_rowid") if key in frame]
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


def _qc_metadata(manifest, period=None):
    """Validate QC v1 manifest fields and return the reader metadata they imply."""
    if manifest.get("status") != "complete" or manifest.get("continuity_violations") != 0:
        raise ValueError("QC output is not recorded as complete with zero continuity violations")
    if period is not None and manifest.get("period") != period:
        raise ValueError(f"QC metadata period {manifest.get('period')!r} differs from {period!r}")
    protocol = manifest.get("protocol_id")
    if protocol not in QC_PROTOCOLS or manifest.get("protocol_sha256") != QC_PROTOCOLS[protocol]:
        raise ValueError(f"Unsupported QC protocol {protocol!r}; expected LiMOSAT QC v1")
    metadata = {
        "period": manifest["period"], "crs": QC_CRS, "qc_protocol_id": protocol,
        "qc_protocol_sha256": manifest["protocol_sha256"], "qc_flag_bits": dict(QC_FLAG_BITS),
    }
    if manifest["period"] in OWNERSHIP:
        start, stop = OWNERSHIP[manifest["period"]]
        metadata["ownership"] = {"policy_id": OWNERSHIP_POLICY, "start": start, "stop_exclusive": stop}
    return metadata


def _link_types(frame):
    for name in ("source_image_id", "target_image_id", "source_orbit_num", "target_orbit_num", "source_converged_to", "target_converged_to"):
        if name in frame:
            frame[name] = frame[name].astype("Int64")
    for name in ("local_supported", "direct"):
        if name in frame:
            frame[name] = frame[name].astype("boolean")
    for name in ("start_time_utc", "end_time_utc"):
        if name in frame:
            frame[name] = pd.to_datetime(frame[name], utc=True, format="ISO8601", errors="raise").astype("datetime64[ns, UTC]")
    return frame


class LimosatDataset:
    """A single source period, with explicit point and displacement views.

    Dates are half-open UTC intervals. ``max_rows`` is a preview limit, not a
    statistically representative sample. Connections never modify input files.
    """

    def __init__(self, path, *, table=None, crs=None):
        self.path = Path(path).expanduser().resolve()
        self.table = None
        self._links = None
        self._kept_links = None
        self.metadata = {"source_path": str(self.path), "crs": None, "qc_protocol_id": None}
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        suffix = self.path.suffix.lower()
        if self.path.is_dir():
            raise ValueError("Choose one period file; combining periods needs an explicit ownership policy")
        if table is not None and suffix not in {".sqlite", ".db"}:
            raise ValueError("table applies only to SQLite")
        if suffix in {".sqlite", ".db"}:
            self._inspect_sqlite(table)
        elif suffix == ".parquet":
            self._inspect_parquet()
        elif suffix == ".feather":
            self.kind = "feather_points"
            self.metadata["legacy_quality_assumption"] = "q is interpreted as historical correlation"
        else:
            raise ValueError("Expected a SQLite, Parquet or Feather file")
        known_crs = self.metadata["crs"]
        if crs is not None and known_crs is not None and crs.upper() != known_crs.upper():
            raise ValueError(f"Source CRS is {known_crs}; crs does not reproject coordinates")
        self.metadata["crs"] = known_crs or crs
        if self.metadata["crs"] is None:
            raise ValueError("Source has no CRS metadata; supply crs explicitly")
        self.metadata.setdefault("period", self.path.stem)
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
                if "qc_metadata" not in names or "qc_removed_links" not in names:
                    raise ValueError("QC table has no qc_metadata/qc_removed_links; only LiMOSAT QC v1 SQLite is supported")
                manifest = {key: json.loads(value) for key, value in conn.execute("SELECT key, value_json FROM qc_metadata")}
                self.metadata.update(_qc_metadata(manifest, table.removesuffix("__qc")))
                if self.path.with_suffix(".parquet").is_file():
                    self.metadata["original_trajectory_ids"] = "from kept and removed links by rowid; missing for observations in no link"
                else:
                    self.metadata["original_trajectory_ids"] = "unavailable: no kept-link Parquet beside the SQLite"
            else:
                self.metadata["period"] = table

    def _inspect_parquet(self):
        schema = pq.ParquetFile(self.path).schema_arrow
        fields = set(schema.names)
        if "qc_exclusion" in fields:
            raise ValueError("This is a removed-link file; open the kept-link Parquet and use read_removed_links()")
        missing = set(QC_LINK_COLUMNS) - fields
        if missing:
            raise ValueError(f"Expected the LiMOSAT QC v1 link schema; missing {sorted(missing)}")
        for name in ("start_time_utc", "end_time_utc"):
            dtype = schema.field(name).type
            if not pa.types.is_timestamp(dtype) or dtype.tz != "UTC":
                raise ValueError("Expected UTC timestamp columns in QC v1 Parquet")
        if not pa.types.is_integer(schema.field("qc_flags").type):
            raise ValueError("Expected integer qc_flags bits in QC v1 Parquet")
        sidecar = self.path.with_suffix(".manifest.json")
        if not sidecar.is_file():
            raise ValueError(f"QC v1 Parquet requires its manifest: {sidecar.name}")
        self.metadata.update(_qc_metadata(json.loads(sidecar.read_text())))
        arrow = json.loads((schema.metadata or {}).get(b"limosat.qc", b"{}"))
        if arrow.get("protocol_id") != self.metadata["qc_protocol_id"] or arrow.get("flag_bits") != QC_FLAG_BITS:
            raise ValueError("Parquet limosat.qc metadata disagrees with the manifest protocol or flag bits")
        self.kind = "parquet_qc_links"
        self._links = ds.dataset(self.path, format="parquet")
        self._link_fields = fields

    def _original_ids(self, rowids):
        """Map observation rowids to original trajectories through QC links.

        Observations in no kept or removed link remain missing; the QC
        trajectory ID is never substituted.
        """
        original = pd.array(np.full(len(rowids), pd.NA), dtype="Int64")
        kept_path = self.path.with_suffix(".parquet")
        if not kept_path.is_file() or rowids.empty:
            return original
        if self._kept_links is None:
            kept = LimosatDataset(kept_path)
            if any(kept.metadata[key] != self.metadata[key] for key in ("period", "qc_protocol_sha256")):
                raise ValueError("Kept-link Parquet belongs to a different period or QC protocol")
            self._kept_links = kept
        wanted = pa.array(rowids.unique().astype("int64"))
        columns = ["source_rowid", "target_rowid", "original_trajectory_id"]
        expression = ds.field("source_rowid").isin(wanted) | ds.field("target_rowid").isin(wanted)
        kept = self._kept_links._links.to_table(columns=columns, filter=expression).to_pandas()
        with _connect(self.path) as conn:
            removed = pd.read_sql_query(f"SELECT {', '.join(columns)} FROM qc_removed_links", conn)
        pairs = pd.concat([
            frame[[f"{end}_rowid", "original_trajectory_id"]].set_axis(["rowid", "original"], axis=1)
            for frame in (kept, removed) for end in ("source", "target")
        ]).drop_duplicates()
        if pairs.rowid.duplicated().any():
            raise ValueError("An observation maps to conflicting original trajectory IDs")
        lookup = pairs.set_index("rowid").original.astype("Int64")
        return lookup.reindex(rowids.to_numpy(dtype="int64")).array

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
        if self.metadata["qc_protocol_id"]:
            frame["qc_trajectory_id"] = frame.trajectory_id
            frame["original_trajectory_id"] = self._original_ids(frame.point_rowid)
        frame["period"] = self.metadata.get("period", self.path.stem)
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
        if self._links is not None:
            if not linked_only:
                raise ValueError("QC link Parquet excludes singleton observations; use linked_only=True or the QC SQLite for complete points")
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

    def _read_parquet(self, start, stop, trajectory_ids, *, endpoint_times=False, columns=None, max_rows=None, flag_mask=0):
        expression = _time_expression("start_time_utc", start, stop)
        if endpoint_times:
            expression |= _time_expression("end_time_utc", start, stop)
        if trajectory_ids is not None:
            expression &= ds.field("qc_trajectory_id").isin([int(value) for value in trajectory_ids])
        if flag_mask:
            expression &= pc.equal(pc.bit_wise_and(ds.field("qc_flags"), pa.scalar(flag_mask, pa.uint16())), 0)
        scan_columns = None
        if columns is not None:
            missing = set(columns) - self._link_fields - {"trajectory_id", "period"}
            if missing:
                raise ValueError(f"Unavailable columns: {sorted(missing)}")
            # Keep ordering independent of output projection.
            needed = set(columns) & self._link_fields
            needed |= {"qc_trajectory_id", "trajectory_link_index", "start_time_utc", "source_image_id", "source_rowid"}
            scan_columns = sorted(needed)
        scanner = self._links.scanner(filter=expression, columns=scan_columns, batch_size=100_000)
        frame = scanner.to_table().to_pandas() if max_rows is None else scanner.head(max_rows).to_pandas()
        frame["trajectory_id"] = frame.qc_trajectory_id.astype("int64")
        frame["period"] = self.metadata["period"]
        return _link_types(frame)

    def _endpoints(self, links):
        frames = []
        for prefix, endpoint, time in (("source", "0", "start_time_utc"), ("target", "1", "end_time_utc")):
            mapping = {f"{prefix}_image_id": "image_id", time: "time_utc", f"x{endpoint}_m": "x_m", f"y{endpoint}_m": "y_m"}
            for name in ENDPOINT_FIELDS:
                mapping[f"{prefix}_{name}"] = name
            mapping[f"{prefix}_rowid"] = "point_rowid"
            shared = [field for field in ("trajectory_id", "qc_trajectory_id", "original_trajectory_id", "period") if field in links]
            fields = [field for field in mapping if field in links]
            frames.append(links[shared + fields].rename(columns=mapping))
        frame = pd.concat(frames, ignore_index=True)
        for name in ("image_id", "point_rowid"):
            if name in frame:
                frame[name] = frame[name].astype("Int64")
        keys = [key for key in ("period", "trajectory_id") if key in frame]
        keys += ["point_rowid"] if "point_rowid" in frame else ["image_id", "time_utc"]
        # An internal point occurs once as a target and once as a source.
        state = [field for field in ("image_id", "time_utc", "x_m", "y_m", *ENDPOINT_FIELDS) if field in frame and field not in keys]
        if frame.drop_duplicates(keys + state).duplicated(keys).any():
            raise ValueError("Displacement endpoints disagree about a shared observation")
        return frame.drop_duplicates(keys).copy()

    def _owned(self, start, stop):
        ownership = self.metadata.get("ownership")
        if not ownership:
            raise ValueError("Source has no release ownership interval")
        owner_start, owner_stop = _bounds(ownership["start"], ownership["stop_exclusive"])
        start = max(start, owner_start) if start is not None else owner_start
        stop = min(stop, owner_stop) if stop is not None else owner_stop
        # An empty overlap still produces a typed result through the read path.
        return min(start, stop), stop

    def read_displacements(self, *, start=None, stop=None, trajectory_ids=None, columns=None, owned_only=False, exclude_flags=(), max_rows=None):
        """Read consecutive links, filtered by their start time after pairing.

        QC boundaries are preserved. No correlation, speed or interpolation
        threshold is applied. Non-positive elapsed times yield missing speed.
        ``exclude_flags`` names QC v1 flags (e.g. ``"LOW_PAIR_DIRECT_SHARE"``)
        whose links are dropped; by default every kept link is returned.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        if isinstance(exclude_flags, str):
            exclude_flags = (exclude_flags,)
        unknown = set(exclude_flags) - set(QC_FLAG_BITS)
        if unknown:
            raise ValueError(f"Unknown QC flags: {sorted(unknown)}; expected {sorted(QC_FLAG_BITS)}")
        if exclude_flags and self._links is None:
            raise ValueError("QC flags are stored only in the QC v1 link Parquet")
        if owned_only:
            start, stop = self._owned(start, stop)
        if self._links is not None:
            flag_mask = sum(1 << QC_FLAG_BITS[name] for name in set(exclude_flags))
            frame = self._read_parquet(start, stop, trajectory_ids, columns=columns, max_rows=max_rows, flag_mask=flag_mask)
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
        metadata = {**self.metadata, "ownership_applied": owned_only, "excluded_flags": sorted(set(exclude_flags)), "time_filter": "displacement start time"}
        return _finish(frame, columns, metadata, max_rows)

    def read_removed_links(self, *, start=None, stop=None, original_trajectory_ids=None, columns=None, owned_only=False, max_rows=None):
        """Read links removed by QC v1, with ``qc_exclusion`` and diagnostics.

        Removed links have no QC trajectory; select them by original ID. The
        QC SQLite stores them in ``qc_removed_links``; link Parquet reads the
        ``<period>_qc_removed.parquet`` file beside it.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        if not self.metadata["qc_protocol_id"]:
            raise ValueError("Removed links are available only from LiMOSAT QC v1 outputs")
        if owned_only:
            start, stop = self._owned(start, stop)
        if self._links is not None:
            path = self.path.with_name(self.path.stem + "_removed.parquet")
            if not path.is_file():
                raise FileNotFoundError(path)
            fields = set(pq.ParquetFile(path).schema_arrow.names)
            if not set(QC_REMOVED_COLUMNS) <= fields:
                raise ValueError("Expected the LiMOSAT QC v1 removed-link schema")
            expression = _time_expression("start_time_utc", start, stop)
            if original_trajectory_ids is not None:
                expression &= ds.field("original_trajectory_id").isin([int(value) for value in original_trajectory_ids])
            scan_columns = None
            if columns is not None:
                missing = set(columns) - fields - {"period"}
                if missing:
                    raise ValueError(f"Unavailable columns: {sorted(missing)}")
                scan_columns = sorted(set(columns) & fields | {"original_trajectory_id", "trajectory_link_index", "start_time_utc", "source_image_id", "source_rowid"})
            scanner = ds.dataset(path, format="parquet").scanner(filter=expression, columns=scan_columns, batch_size=100_000)
            frame = scanner.to_table().to_pandas() if max_rows is None else scanner.head(max_rows).to_pandas()
        else:
            where, params = [], []
            # Stored as 'YYYY-MM-DD HH:MM:SS[.ffffff]+00:00' text. Whole-second
            # bounds select a superset, refined exactly after parsing.
            if start is not None:
                where.append("start_time_utc >= ?")
                params.append(start.floor("s").strftime("%Y-%m-%d %H:%M:%S"))
            if stop is not None:
                where.append("start_time_utc < ?")
                params.append(stop.ceil("s").strftime("%Y-%m-%d %H:%M:%S"))
            if original_trajectory_ids is not None:
                where.append("original_trajectory_id IN (SELECT CAST(value AS INTEGER) FROM json_each(?))")
                params.append(json.dumps([int(value) for value in original_trajectory_ids]))
            query = "SELECT * FROM qc_removed_links" + (" WHERE " + " AND ".join(where) if where else "")
            if max_rows is not None:
                query += " LIMIT ?"
                params.append(max_rows)
            with _connect(self.path) as conn:
                frame = pd.read_sql_query(query, conn, params=params)
        frame["period"] = self.metadata["period"]
        frame = _link_types(frame)
        frame = frame.loc[_time_mask(frame.start_time_utc, start, stop)].copy()
        metadata = {**self.metadata, "ownership_applied": owned_only, "time_filter": "displacement start time", "link_coverage": "links removed by QC"}
        return _finish(frame, columns, metadata, max_rows)

    def _pair_points(self, points):
        keys = ["period", "trajectory_id"]
        following = points.groupby(keys, sort=False).shift(-1)
        keep = following.time_utc.notna()
        if "is_last" in points:
            keep &= points.is_last.eq(0)
        source, target = points.loc[keep], following.loc[keep]
        shared = [field for field in ("period", "trajectory_id", "qc_trajectory_id", "original_trajectory_id") if field in points]
        frame = source[shared].copy()
        for prefix, endpoint, time in (("source", "0", "start_time_utc"), ("target", "1", "end_time_utc")):
            values = source if prefix == "source" else target
            frame[f"{prefix}_image_id"] = values.image_id.astype("Int64")
            frame[time] = values.time_utc
            frame[f"x{endpoint}_m"], frame[f"y{endpoint}_m"] = values.x_m, values.y_m
            for field in ENDPOINT_FIELDS:
                if field in values:
                    frame[f"{prefix}_{field}"] = values[field]
            if "point_rowid" in values:
                frame[f"{prefix}_rowid"] = values.point_rowid.astype("int64")
        frame["elapsed_seconds"] = (frame.end_time_utc - frame.start_time_utc).dt.total_seconds()
        frame["positive_elapsed"] = frame.elapsed_seconds > 0
        frame["dx_m"] = frame.x1_m - frame.x0_m
        frame["dy_m"] = frame.y1_m - frame.y0_m
        frame["distance_m"] = np.hypot(frame.dx_m, frame.dy_m)
        frame["speed_m_per_day"] = frame.distance_m / frame.elapsed_seconds.where(frame.positive_elapsed) * 86_400
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
