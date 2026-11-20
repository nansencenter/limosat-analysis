"""Reader for the public LiMOSAT trajectory release: one NetCDF4 file (CF trajectory, contiguous ragged array) per period.

One file holds one processing period. ``trajectory`` and ``obs`` are the two dimensions: ``rowsize`` gives the number
of observations of each trajectory, trajectories are stored in order of their start time (``time_start``) and the
observations of a trajectory in ``point_index`` order. A time window therefore reads as a few contiguous runs and one
trajectory as a single slice. Every column is its own variable, so only the requested columns are read and decompressed.

The reader needs ``netCDF4``. All metadata (schema version, flag names, primary window, descriptions) comes from the file.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import (CORE_COLUMNS, CORE_DISPLACEMENT_COLUMNS, DISPLACEMENT_COLUMNS, SUPPORTED_MAJOR_VERSION,
                      bounds as _bounds, limit as _limit, resolve_columns)

import netCDF4

OBS_COLUMNS = ("point_index", "time", "scene", "x_m", "y_m", "lon", "lat", "dx_m", "dy_m", "speed_m_per_day", "interpolated",
               "correlation", "direct_match_share", "quality_flags", "is_primary")
_SCENE = "scene_index"


def is_point_file(path) -> bool:
    """True if ``path`` is a NetCDF4 file written in the public LiMOSAT format."""
    path = Path(path).expanduser()
    if path.suffix != ".nc" or not path.is_file():
        return False
    try:
        with netCDF4.Dataset(path) as d:
            return "schema_version" in d.ncattrs() and getattr(d, "featureType", "") == "trajectory"
    except OSError:
        return False


class LimosatTrajectories:
    """A public LiMOSAT trajectory file (NetCDF4): the trajectories of one processing period."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.kind = "netcdf_ragged"
        self._index = None
        self._scene_names = None
        # The file is opened for each read and closed again: no handle is kept, so objects need no clean-up.
        with self._dataset() as d:
            attrs = set(d.ncattrs())
            if "schema_version" not in attrs or getattr(d, "featureType", "") != "trajectory":
                raise ValueError("Not a LiMOSAT trajectory file: no schema_version / featureType trajectory attributes")
            if int(d.schema_version.split(".")[0]) != SUPPORTED_MAJOR_VERSION:
                raise ValueError(f"Unsupported LiMOSAT schema {d.schema_version}; this reader supports {SUPPORTED_MAJOR_VERSION}.x")
            flags = d["quality_flags"]
            masks = [int(m) for m in np.atleast_1d(flags.flag_masks)]
            names = flags.flag_meanings.split()
            self.metadata = {
                "schema_version": d.schema_version,
                "release_id": getattr(d, "release_id", None),
                "period": getattr(d, "period", None),
                "crs": getattr(d["crs"], "epsg_code", None) if "crs" in d.variables else None,
                "flag_bits": {name: int(np.log2(mask)) for name, mask in zip(names, masks)},
                "primary_window": {"start": d.primary_window_start, "stop_exclusive": d.primary_window_stop_exclusive},
                "primary_policy": getattr(d, "primary_rule", None),
                "rows": int(d.dimensions["obs"].size),
                "trajectories": int(d.dimensions["trajectory"].size),
            }
            self._gap = (d["time"].chunking() or [65_536])[0]

    @contextmanager
    def _dataset(self):
        d = netCDF4.Dataset(self.path)
        d.set_auto_maskandscale(False)
        try:
            yield d
        finally:
            d.close()

    def __repr__(self):
        return (f"LimosatTrajectories({self.path.name!r}, period={self.metadata['period']!r}, rows={self.metadata['rows']:,}, "
                f"trajectories={self.metadata['trajectories']:,})")

    def close(self):
        """Nothing to close: the file is opened only while a read runs. Kept so that ``with open_limosat(...)`` works."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    # ----------------------------------------------------------------------------------------- helpers

    def describe(self) -> pd.DataFrame:
        """Variable name, type, unit and description, as recorded in the file."""
        rows = []
        with self._dataset() as d:
            for name in ("trajectory_id", *OBS_COLUMNS):
                variable = d[_SCENE if name == "scene" else name]
                rows.append({"column": name, "type": str(variable.dtype), "unit": getattr(variable, "units", ""),
                             "description": getattr(d["scene_name"] if name == "scene" else variable, "long_name", "")})
        return pd.DataFrame(rows)

    def _traj(self):
        if self._index is None:
            with self._dataset() as d:
                rowsize = d["rowsize"][:].astype(np.int64)
                self._index = dict(ids=d["trajectory_id"][:], rowsize=rowsize, tstart=d["time_start"][:], tend=d["time_end"][:],
                                   offset=np.r_[0, np.cumsum(rowsize)])
        return self._index

    def _names(self):
        if self._scene_names is None:
            with self._dataset() as d:
                self._scene_names = np.array(list(d["scene_name"][:]), dtype=object)
        return self._scene_names

    def _runs(self, selected):
        """Contiguous (first trajectory, last trajectory + 1) groups of the selected trajectory indices, merging small gaps."""
        sel = np.asarray(selected, dtype=np.int64)
        if len(sel) == 0:
            return []
        offset = self._traj()["offset"]
        breaks = np.flatnonzero(offset[sel[1:]] - offset[sel[:-1] + 1] > self._gap)
        firsts = np.r_[sel[0], sel[breaks + 1]]
        lasts = np.r_[sel[breaks], sel[-1]] + 1
        return list(zip(firsts.tolist(), lasts.tolist()))

    def _read(self, selected, names) -> pd.DataFrame:
        """All observations of the trajectories ``selected`` (sorted indices), as a frame with ``trajectory_id`` and ``names``."""
        t = self._traj()
        selected = np.asarray(selected, dtype=np.int64)
        names = [n for n in dict.fromkeys(names) if n != "trajectory_id"]
        parts = {n: [] for n in names}
        ids = []
        wanted = np.zeros(len(t["ids"]), dtype=bool)
        wanted[selected] = True
        with self._dataset() as d:
            for a, b in self._runs(selected):
                lo, hi = t["offset"][a], t["offset"][b]
                keep = np.repeat(wanted[a:b], t["rowsize"][a:b])
                everything = bool(keep.all())
                ids.append(np.repeat(t["ids"][a:b], t["rowsize"][a:b])[keep])
                for n in names:
                    values = d[_SCENE if n == "scene" else n][lo:hi]
                    parts[n].append(values if everything else values[keep])
        out = {"trajectory_id": np.concatenate(ids) if ids else np.array([], dtype=np.int64)}
        for n in names:
            out[n] = np.concatenate(parts[n]) if parts[n] else np.array([])
        return self._decode(out)

    def _read_rows(self, selected, names, start=None, stop=None, primary_only=False) -> dict:
        """Observations of the candidate trajectories ``selected`` that fall in [start, stop), as raw arrays plus ``trajectory_id``.

        ``time`` (and ``is_primary``) are read first for the whole candidate runs; the other variables are then read only for
        the runs of rows that are wanted, so a window that covers a small part of long trajectories decompresses little.
        """
        t = self._traj()
        selected = np.asarray(selected, dtype=np.int64)
        names = [n for n in dict.fromkeys(names) if n != "trajectory_id"]
        wanted = np.zeros(len(t["ids"]), dtype=bool)
        wanted[selected] = True
        parts = []
        with self._dataset() as d:
            for a, b in self._runs(selected):
                lo, hi = t["offset"][a], t["offset"][b]
                keep = np.repeat(wanted[a:b], t["rowsize"][a:b])
                if start is not None or stop is not None:
                    tt = d["time"][lo:hi]
                    if start is not None:
                        keep &= tt >= start.timestamp()
                    if stop is not None:
                        keep &= tt < stop.timestamp()
                if primary_only:
                    keep &= d["is_primary"][lo:hi] == 1
                parts.append(lo + np.flatnonzero(keep))
            rows = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
            out = {n: [] for n in names}
            if len(rows):
                breaks = np.flatnonzero(np.diff(rows) > self._gap)
                first = np.r_[0, breaks + 1]
                last = np.r_[breaks + 1, len(rows)]
                for i, j in zip(first.tolist(), last.tolist()):
                    lo, hi = int(rows[i]), int(rows[j - 1]) + 1
                    contiguous = hi - lo == j - i
                    local = None if contiguous else rows[i:j] - lo
                    for n in names:
                        values = d[_SCENE if n == "scene" else n][lo:hi]
                        out[n].append(values if contiguous else values[local])
        raw = {n: (np.concatenate(v) if v else np.array([])) for n, v in out.items()}
        raw["trajectory_id"] = t["ids"][np.searchsorted(t["offset"], rows, side="right") - 1]
        return raw

    def _decode(self, raw) -> pd.DataFrame:
        frame = {}
        for name, values in raw.items():
            if name == "time":
                frame[name] = pd.to_datetime(np.rint(values * 1e3).astype("int64"), unit="ms", utc=True).astype("datetime64[ns, UTC]")
            elif name == "scene":
                frame[name] = pd.Categorical.from_codes(values, categories=self._names())
            elif name == "interpolated":
                frame[name] = pd.array(np.where(values < 0, 0, values).astype(bool), dtype="boolean")
                frame[name][values < 0] = pd.NA
            elif name == "quality_flags":
                frame[name] = pd.array(np.where(values == 255, 0, values), dtype="UInt8")
                frame[name][values == 255] = pd.NA
            elif name == "is_primary":
                frame[name] = pd.array(values.astype(bool), dtype="boolean")
            else:
                frame[name] = values
        return pd.DataFrame(frame)

    def _candidates(self, start, stop):
        """Indices of trajectories that overlap [start, stop) in time (all of them without bounds)."""
        t = self._traj()
        keep = np.ones(len(t["ids"]), dtype=bool)
        if start is not None:
            keep &= t["tend"] >= start.timestamp()
        if stop is not None:
            keep &= t["tstart"] < stop.timestamp()
        return np.flatnonzero(keep)

    def _index_of(self, trajectory_ids):
        t = self._traj()
        wanted = np.atleast_1d(trajectory_ids).astype(np.int64)
        return np.flatnonzero(np.isin(t["ids"], wanted))

    def _attrs(self, frame):
        frame.attrs.update(self.metadata)
        return frame

    # ----------------------------------------------------------------------------------------- readers

    def read_points(self, *, start=None, stop=None, trajectory_ids=None, columns=None, primary_only=False,
                    max_rows=None) -> pd.DataFrame:
        """Observations with ``start <= time < stop`` (UTC; naive dates are UTC), sorted by time (ties keep the file order: trajectory start time, then ``point_index``).

        ``columns`` is None for the core columns (``CORE_COLUMNS``), ``"all"`` for every variable, or a list of names.
        ``primary_only`` keeps rows in the period's own window. ``trajectory_ids`` are unique within this period only.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        columns = resolve_columns(columns, ("trajectory_id", *OBS_COLUMNS), CORE_COLUMNS)
        selected = self._candidates(start, stop)
        if trajectory_ids is not None:
            selected = np.intersect1d(selected, self._index_of(trajectory_ids))
        raw = self._read_rows(selected, [*columns, "time"], start, stop, primary_only)
        order = np.argsort(raw["time"], kind="stable")  # ties keep the file order: trajectory start time, then point_index
        if max_rows is not None:
            order = order[:max_rows]
        frame = self._decode({name: values[order] for name, values in raw.items()})
        return self._attrs(frame[columns].reset_index(drop=True))

    def read_trajectories(self, *, start=None, stop=None, xlim=None, ylim=None, trajectory_ids=None, columns=None,
                          max_rows=None) -> pd.DataFrame:
        """Complete trajectories: every observation of each trajectory that has an observation in the window.

        ``start``/``stop`` select by observation time and ``xlim``/``ylim`` (EPSG:3413 metres) by position; an
        observation inside both makes the whole trajectory part of the result. The result is ordered by trajectory
        start time and ``point_index``, so ``groupby('trajectory_id')`` yields one track at a time. Give
        ``trajectory_ids`` alone to read specific trajectories.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        columns = resolve_columns(columns, ("trajectory_id", *OBS_COLUMNS), CORE_COLUMNS)
        if start is not None or stop is not None or xlim is not None or ylim is not None:
            selected = self._candidates(start, stop)
            if trajectory_ids is not None:
                selected = np.intersect1d(selected, self._index_of(trajectory_ids))
            probe = self._decode(self._read_rows(selected, ["time", "x_m", "y_m"], start, stop))
            hit = np.ones(len(probe), dtype=bool)
            if xlim is not None:
                hit &= ((probe["x_m"] >= xlim[0]) & (probe["x_m"] <= xlim[1])).to_numpy()
            if ylim is not None:
                hit &= ((probe["y_m"] >= ylim[0]) & (probe["y_m"] <= ylim[1])).to_numpy()
            selected = self._index_of(np.unique(probe["trajectory_id"].to_numpy()[hit]))
        elif trajectory_ids is not None:
            selected = self._index_of(trajectory_ids)
        else:
            raise ValueError("Give start/stop, xlim/ylim or trajectory_ids; to read everything use read_points()")
        frame = self._read(selected, [*columns, "point_index"])
        frame = frame[columns]
        return self._attrs(frame.iloc[:max_rows] if max_rows is not None else frame)

    def read_displacements(self, *, start=None, stop=None, trajectory_ids=None, columns=None, primary_only=False,
                           exclude_flags=(), max_rows=None) -> pd.DataFrame:
        """One row per displacement between consecutive observations of a trajectory.

        ``start <= start_time < stop`` selects by the time the displacement starts, which is also what ``is_primary`` is
        defined on. Columns: ``trajectory_id``, ``point_index`` (of the end observation), ``start_time``, ``end_time``,
        ``elapsed_seconds``, ``x0_m``, ``y0_m``, ``x1_m``, ``y1_m``, the displacement columns, ``start_scene``/``end_scene``,
        ``is_primary``, ``lon``/``lat`` of the end observation. ``columns`` is None for ``CORE_DISPLACEMENT_COLUMNS``,
        ``"all"`` or a list of names. ``exclude_flags`` names flags (see ``metadata['flag_bits']``) whose displacements
        are dropped.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        if isinstance(exclude_flags, str):
            exclude_flags = (exclude_flags,)
        bits = self.metadata["flag_bits"]
        unknown = set(exclude_flags) - set(bits)
        if unknown:
            raise ValueError(f"Unknown flags: {sorted(unknown)}; this file declares {sorted(bits)}")
        available = ("trajectory_id", "point_index", "start_time", "end_time", "elapsed_seconds", "x0_m", "y0_m", "x1_m",
                     "y1_m", *DISPLACEMENT_COLUMNS, "start_scene", "end_scene", "is_primary", "lon", "lat")
        columns = resolve_columns(columns, available, CORE_DISPLACEMENT_COLUMNS)
        need = ["time", "point_index", "x_m", "y_m"]
        for name in columns:
            if name in DISPLACEMENT_COLUMNS or name in ("is_primary", "lon", "lat"):
                need.append(name)
        if any(c in columns for c in ("start_scene", "end_scene")):
            need.append("scene")
        if primary_only:
            need.append("is_primary")
        if exclude_flags:
            need.append("quality_flags")
        selected = self._candidates(start, stop)
        if trajectory_ids is not None:
            selected = np.intersect1d(selected, self._index_of(trajectory_ids))
        rows = self._read(selected, need)
        previous = rows.groupby("trajectory_id", sort=False).shift(1)
        has_previous = previous["time"].notna().to_numpy()
        end, before = rows[has_previous], previous[has_previous]
        frame = pd.DataFrame({"trajectory_id": end["trajectory_id"], "point_index": end["point_index"],
                              "start_time": before["time"], "end_time": end["time"],
                              "elapsed_seconds": (end["time"] - before["time"]).dt.total_seconds(),
                              "x0_m": before["x_m"], "y0_m": before["y_m"], "x1_m": end["x_m"], "y1_m": end["y_m"]})
        for name in (*DISPLACEMENT_COLUMNS, "is_primary", "lon", "lat"):
            if name in end:
                frame[name] = end[name]
        if "scene" in end:
            frame["start_scene"], frame["end_scene"] = before["scene"], end["scene"]
        keep = pd.Series(True, index=frame.index)
        if start is not None:
            keep &= frame["start_time"] >= start
        if stop is not None:
            keep &= frame["start_time"] < stop
        if primary_only:
            keep &= frame["is_primary"].fillna(False).astype(bool)
        if exclude_flags:
            mask = sum(1 << bits[name] for name in set(exclude_flags))
            keep &= (frame["quality_flags"].fillna(0).astype("int64") & mask) == 0
        frame = frame[keep].sort_values(["start_time", "trajectory_id", "point_index"], kind="stable")
        frame.attrs["excluded_flags"] = sorted(set(exclude_flags))
        if max_rows is not None:
            frame = frame.iloc[:max_rows]
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


def open_limosat(path) -> LimosatTrajectories:
    """Open one period file of the public LiMOSAT trajectory release (``limosat_<period>.nc``).

    Reads return the core columns (``CORE_COLUMNS``) unless ``columns`` is a list of names or ``"all"``. The methods are
    ``read_points``, ``read_trajectories``, ``read_displacements``, ``describe`` and ``verify``.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise ValueError("Open one period file, limosat_<period>.nc; periods are not combined automatically")
    if not is_point_file(path):
        raise ValueError(
            f"{path.name} is not a LiMOSAT trajectory file (NetCDF4 with schema_version and featureType trajectory attributes). "
            "Raw tracking databases, the earlier benchmark Feather files, the internal QC v1 archive and the Parquet versions of "
            "the release are not read by this version; use limosat-analysis 0.2.0 for the Parquet release and commit 6d3694b for "
            "the older sources (pandas.read_feather reads the Feather files)."
        )
    return LimosatTrajectories(path)
