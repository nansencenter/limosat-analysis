"""Reader for the public LiMOSAT trajectory files in NetCDF4 (CF trajectory, contiguous ragged array).

One file holds one processing period. ``trajectory`` and ``obs`` are the two dimensions: ``rowsize`` gives the number
of observations of each trajectory, trajectories are stored in order of their start time (``time_start``) and the
observations of a trajectory in ``point_index`` order. A time window therefore reads as a few contiguous runs and one
trajectory as a single slice. Every column is its own variable, so only the requested columns are read and decompressed.

The reader needs ``netCDF4``. All metadata (schema version, flag names, primary window, descriptions) comes from the file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import (CORE_COLUMNS, CORE_DISPLACEMENT_COLUMNS, DISPLACEMENT_COLUMNS, SUPPORTED_MAJOR_VERSION,
                      bounds as _bounds, limit as _limit, resolve_columns)

try:  # the reader module is importable without netCDF4; opening a file is not
    import netCDF4
except ImportError:  # pragma: no cover - environment dependent
    netCDF4 = None

OBS_COLUMNS = ("point_index", "time", "scene", "x_m", "y_m", "lon", "lat", "dx_m", "dy_m", "speed_m_per_day", "interpolated",
               "correlation", "direct_match_share", "quality_flags", "is_primary")
_SCENE = "scene_index"


def is_netcdf_file(path) -> bool:
    """True if ``path`` is a NetCDF4 file written in the public LiMOSAT format."""
    path = Path(path).expanduser()
    if path.suffix != ".nc" or not path.is_file() or netCDF4 is None:
        return False
    try:
        with netCDF4.Dataset(path) as d:
            return "schema_version" in d.ncattrs() and getattr(d, "featureType", "") == "trajectory"
    except OSError:
        return False


class LimosatNetCDF:
    """A public LiMOSAT trajectory file (NetCDF4): the trajectories of one processing period."""

    def __init__(self, path):
        if netCDF4 is None:
            raise ImportError("Reading NetCDF needs netCDF4: pip install netCDF4")
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._ds = netCDF4.Dataset(self.path)
        self._ds.set_auto_maskandscale(False)
        d = self._ds
        attrs = set(d.ncattrs())
        if "schema_version" not in attrs or getattr(d, "featureType", "") != "trajectory":
            raise ValueError("Not a LiMOSAT trajectory file: no schema_version / featureType trajectory attributes")
        if int(d.schema_version.split(".")[0]) != SUPPORTED_MAJOR_VERSION:
            raise ValueError(f"Unsupported LiMOSAT schema {d.schema_version}; this reader supports {SUPPORTED_MAJOR_VERSION}.x")
        self.kind = "netcdf_ragged"
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
        self._index = None
        self._scene_names = None

    def __repr__(self):
        return (f"LimosatNetCDF({self.path.name!r}, period={self.metadata['period']!r}, rows={self.metadata['rows']:,}, "
                f"trajectories={self.metadata['trajectories']:,})")

    def close(self):
        self._ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----------------------------------------------------------------------------------------- helpers

    def describe(self) -> pd.DataFrame:
        """Variable name, type, unit and description, as recorded in the file."""
        rows = []
        for name in ("trajectory_id", *OBS_COLUMNS):
            variable = self._ds[_SCENE if name == "scene" else name]
            rows.append({"column": name, "type": str(variable.dtype), "unit": getattr(variable, "units", ""),
                         "description": getattr(self._ds["scene_name"] if name == "scene" else variable, "long_name", "")})
        return pd.DataFrame(rows)

    def _traj(self):
        if self._index is None:
            d = self._ds
            rowsize = d["rowsize"][:].astype(np.int64)
            self._index = dict(ids=d["trajectory_id"][:], rowsize=rowsize, tstart=d["time_start"][:], tend=d["time_end"][:],
                               offset=np.r_[0, np.cumsum(rowsize)])
        return self._index

    def _names(self):
        if self._scene_names is None:
            self._scene_names = np.array(list(self._ds["scene_name"][:]), dtype=object)
        return self._scene_names

    def _variable(self, name):
        return self._ds[_SCENE if name == "scene" else name]

    def _runs(self, selected):
        """Contiguous (first trajectory, last trajectory + 1) groups of the selected trajectory indices, merging small gaps."""
        if len(selected) == 0:
            return []
        t = self._traj()
        runs, first, last = [], selected[0], selected[0]
        for i in selected[1:]:
            if t["offset"][i] - t["offset"][last + 1] <= self._gap:
                last = i
            else:
                runs.append((first, last + 1))
                first = last = i
        runs.append((first, last + 1))
        return runs

    def _read(self, selected, names) -> pd.DataFrame:
        """All observations of the trajectories ``selected`` (sorted indices), as a frame with ``trajectory_id`` and ``names``."""
        t = self._traj()
        selected = np.asarray(selected, dtype=np.int64)
        names = [n for n in dict.fromkeys(names) if n != "trajectory_id"]
        parts = {n: [] for n in names}
        ids = []
        wanted = np.zeros(len(t["ids"]), dtype=bool)
        wanted[selected] = True
        for a, b in self._runs(selected):
            lo, hi = t["offset"][a], t["offset"][b]
            keep = np.repeat(wanted[a:b], t["rowsize"][a:b])
            everything = bool(keep.all())
            ids.append(np.repeat(t["ids"][a:b], t["rowsize"][a:b])[keep])
            for n in names:
                values = self._variable(n)[lo:hi]
                parts[n].append(values if everything else values[keep])
        out = {"trajectory_id": np.concatenate(ids) if ids else np.array([], dtype=np.int64)}
        for n in names:
            out[n] = np.concatenate(parts[n]) if parts[n] else np.array([])
        return self._decode(out)

    def _decode(self, raw) -> pd.DataFrame:
        frame = {}
        for name, values in raw.items():
            if name == "time":
                frame[name] = pd.to_datetime(np.rint(values * 1e3).astype("int64"), unit="ms", utc=True).astype("datetime64[ns, UTC]")
            elif name == "scene":
                frame[name] = self._names()[values]
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

    @staticmethod
    def _time_mask(frame, start, stop):
        keep = np.ones(len(frame), dtype=bool)
        if start is not None:
            keep &= (frame["time"] >= start).to_numpy()
        if stop is not None:
            keep &= (frame["time"] < stop).to_numpy()
        return keep

    # ----------------------------------------------------------------------------------------- readers

    def read_points(self, *, start=None, stop=None, trajectory_ids=None, columns=None, primary_only=False,
                    max_rows=None) -> pd.DataFrame:
        """Observations with ``start <= time < stop`` (UTC; naive dates are UTC), sorted by time.

        ``columns`` is None for the core columns (``CORE_COLUMNS``), ``"all"`` for every variable, or a list of names.
        ``primary_only`` keeps rows in the period's own window. ``trajectory_ids`` are unique within this period only.
        """
        start, stop = _bounds(start, stop)
        _limit(max_rows)
        columns = resolve_columns(columns, ("trajectory_id", *OBS_COLUMNS), CORE_COLUMNS)
        need = [*columns, "time", "point_index"] + (["is_primary"] if primary_only else [])
        selected = self._candidates(start, stop)
        if trajectory_ids is not None:
            selected = np.intersect1d(selected, self._index_of(trajectory_ids))
        frame = self._read(selected, need)
        keep = self._time_mask(frame, start, stop)
        if primary_only:
            keep &= frame["is_primary"].fillna(False).to_numpy(bool)
        frame = frame[keep].sort_values(["time", "trajectory_id", "point_index"], kind="stable")
        if max_rows is not None:
            frame = frame.iloc[:max_rows]
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
            probe = self._read(selected, ["time", "x_m", "y_m"])
            hit = self._time_mask(probe, start, stop)
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
