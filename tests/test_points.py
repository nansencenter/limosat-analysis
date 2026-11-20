import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import netCDF4
import numpy as np
import pandas as pd
import pytest

from limosat_analysis import CORE_COLUMNS, CORE_DISPLACEMENT_COLUMNS, LimosatTrajectories, open_limosat

PERIOD = "202305_202407"
FLAG_BITS = {"FEW_NEIGHBOURS": 0, "LOW_PAIR_DIRECT_SHARE": 1, "MASK_FALSE_LAND": 2}


def t(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def scene(n):
    return f"S1A_EW_GRDM_1SDH_SCENE{n:02d}"


# trajectory, point_index, time, scene, x, y, dx, dy, speed, interpolated, correlation, share, quality_flags, is_primary
ROWS = [
    (10, 0, "2023-07-30T00:00:00", 1, 0.0, 0.0, None, None, None, None, None, None, None, False),
    (10, 1, "2023-07-31T12:00:00", 2, 100.0, 0.0, 100.0, 0.0, 100 / 1.5, False, 0.5, 0.9, 0, False),
    (10, 2, "2023-08-02T00:00:00", 3, 200.0, 50.0, 100.0, 50.0, 111.80339887498948 / 1.5, True, 0.6, 0.3, 2, False),
    (10, 3, "2023-08-04T00:00:00", 4, 300.0, 50.0, 100.0, 0.0, 50.0, False, 0.7, 0.8, 4, True),
    (11, 0, "2023-09-10T00:00:00", 5, 0.0, 10.0, None, None, None, None, None, None, None, True),
    (11, 1, "2023-09-11T00:00:00", 6, 50.0, 10.0, 50.0, 0.0, 50.0, False, 0.4, 0.6, 1, True),
    (11, 2, "2024-01-01T00:00:00", 7, 50.0, 60.0, 0.0, 50.0, 50 / 112, False, 0.8, 0.7, 0, True),
    (12, 0, "2024-01-01T00:00:00", 7, 5.0, 5.0, None, None, None, None, None, None, None, True),
    (12, 1, "2024-01-03T00:00:00", 8, 25.0, 5.0, 20.0, 0.0, 10.0, False, 0.9, 0.5, 0, True),
]


def write_netcdf(path, rows=ROWS, version="2.0.0"):
    """The observations in the CF contiguous ragged array layout of the public NetCDF files."""
    by_traj = {}
    for row in rows:
        by_traj.setdefault(row[0], []).append(row)
    order = sorted(by_traj, key=lambda k: (t(by_traj[k][0][2]), k))
    obs = [r for k in order for r in sorted(by_traj[k], key=lambda r: r[1])]
    sizes = [len(by_traj[k]) for k in order]
    n = len(obs)
    scenes = sorted({r[3] for r in obs})
    secs = lambda r: t(r[2]).timestamp()
    col = lambda i, dtype="f8": np.array([np.nan if r[i] is None else r[i] for r in obs], dtype=dtype)
    d = netCDF4.Dataset(path, "w", format="NETCDF4")
    d.createDimension("trajectory", len(order)); d.createDimension("obs", n); d.createDimension("scene", len(scenes))
    d.Conventions = "CF-1.8, ACDD-1.3"; d.featureType = "trajectory"; d.schema_version = version; d.release_id = "test-release"
    d.period = PERIOD; d.primary_window_start = "2023-08-01T00:00:00Z"; d.primary_window_stop_exclusive = "2024-08-01T00:00:00Z"
    d.primary_rule = "limosat_release_august_annual_v1"
    d.createVariable("trajectory_id", "i8", ("trajectory",))[:] = order
    v = d.createVariable("rowsize", "i4", ("trajectory",)); v.sample_dimension = "obs"; v[:] = sizes
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    d.createVariable("time_start", "f8", ("trajectory",))[:] = [secs(obs[i]) for i in starts]
    d.createVariable("time_end", "f8", ("trajectory",))[:] = [secs(obs[i + s - 1]) for i, s in zip(starts, sizes)]
    v = d.createVariable("time", "f8", ("obs",), chunksizes=(n,)); v.units = "seconds since 1970-01-01 00:00:00"; v.long_name = "time text"
    v[:] = [secs(r) for r in obs]
    crs = d.createVariable("crs", "i4"); crs.epsg_code = "EPSG:3413"
    d.createVariable("x_m", "f8", ("obs",))[:] = col(4); d.createVariable("y_m", "f8", ("obs",))[:] = col(5)
    d.createVariable("lon", "f8", ("obs",))[:] = col(4); d.createVariable("lat", "f8", ("obs",))[:] = col(5)
    d.createVariable("point_index", "i4", ("obs",))[:] = col(1, "i4")
    for name, i in (("dx_m", 6), ("dy_m", 7), ("speed_m_per_day", 8)):
        v = d.createVariable(name, "f8", ("obs",), fill_value=np.nan)
        v.units = "m" if name != "speed_m_per_day" else "m day-1"; v.long_name = f"{name} text"; v[:] = col(i)
    for name, i in (("correlation", 10), ("direct_match_share", 11)):
        d.createVariable(name, "f4", ("obs",), fill_value=np.float32(np.nan))[:] = col(i, "f4")
    v = d.createVariable("interpolated", "i1", ("obs",), fill_value=np.int8(-1))
    v[:] = [-1 if r[9] is None else int(r[9]) for r in obs]
    v = d.createVariable("quality_flags", "u1", ("obs",), fill_value=np.uint8(255))
    v.flag_masks = np.array([1, 2, 4], dtype=np.uint8); v.flag_meanings = " ".join(FLAG_BITS)
    v[:] = [255 if r[12] is None else r[12] for r in obs]
    d.createVariable("is_primary", "i1", ("obs",))[:] = [int(r[13]) for r in obs]
    d.createVariable("scene_index", "i4", ("obs",))[:] = [scenes.index(r[3]) for r in obs]
    v = d.createVariable("scene_name", str, ("scene",)); v.long_name = "scene text"
    v[:] = np.array([scene(i) for i in scenes], dtype=object)
    d.close()


@pytest.fixture
def path(tmp_path):
    p = tmp_path / f"limosat_{PERIOD}.nc"
    write_netcdf(p)
    return p


@pytest.fixture
def data(path):
    return open_limosat(path)


def test_open_reads_the_file_metadata(data):
    assert isinstance(data, LimosatTrajectories) and data.kind == "netcdf_ragged"
    assert data.metadata["period"] == PERIOD and data.metadata["crs"] == "EPSG:3413"
    assert data.metadata["flag_bits"] == FLAG_BITS and data.metadata["rows"] == 9 and data.metadata["trajectories"] == 3
    assert data.metadata["primary_window"]["stop_exclusive"] == "2024-08-01T00:00:00Z"
    assert "trajectories=3" in repr(data)
    description = data.describe().set_index("column")
    assert description.loc["x_m", "unit"] == "" and "text" in description.loc["speed_m_per_day", "description"]


def test_read_points_filters_and_sorts_by_time(data):
    frame = data.read_points(columns="all")
    assert len(frame) == 9 and frame["time"].is_monotonic_increasing
    assert str(frame["time"].dtype) == "datetime64[ns, UTC]" and str(frame["interpolated"].dtype) == "boolean"
    assert frame.attrs["period"] == PERIOD
    window = data.read_points(start="2023-08-01", stop="2023-09-12")
    assert window[["trajectory_id", "point_index"]].values.tolist() == [[10, 2], [10, 3], [11, 0], [11, 1]]
    assert len(data.read_points(trajectory_ids=[11, 12])) == 5
    assert len(data.read_points(primary_only=True)) == 6
    small = data.read_points(columns=["trajectory_id", "time"], max_rows=2)
    assert list(small) == ["trajectory_id", "time"] and len(small) == 2
    with pytest.raises(ValueError, match="Unavailable"):
        data.read_points(columns=["speed_m_s"])
    with pytest.raises(ValueError, match="positive integer"):
        data.read_points(max_rows=0)
    with pytest.raises(ValueError, match="start < stop"):
        data.read_points(start="2024-01-01", stop="2023-01-01")


def test_reads_return_the_core_columns_unless_asked_otherwise(data):
    assert tuple(data.read_points()) == CORE_COLUMNS
    assert tuple(data.read_trajectories(trajectory_ids=[10])) == CORE_COLUMNS
    assert tuple(data.read_displacements()) == CORE_DISPLACEMENT_COLUMNS
    everything = data.read_points(columns="all")
    assert set(CORE_COLUMNS) < set(everything) and {"scene", "lon", "lat", "correlation", "quality_flags", "interpolated"} <= set(everything)
    assert list(data.read_points(columns="x_m")) == ["x_m"]
    assert list(data.read_displacements(columns=["start_scene", "end_scene"])) == ["start_scene", "end_scene"]


def test_null_displacement_values_on_first_observations(data):
    frame = data.read_points(columns="all").sort_values(["trajectory_id", "point_index"])
    first = frame[frame.point_index == 0]
    assert first[["dx_m", "dy_m", "speed_m_per_day", "correlation", "direct_match_share"]].isna().all().all()
    assert first["interpolated"].isna().all() and first["quality_flags"].isna().all()
    assert frame[frame.point_index > 0][["dx_m", "interpolated", "quality_flags"]].notna().all().all()
    assert str(frame["quality_flags"].dtype) == "UInt8" and frame.loc[(frame.trajectory_id == 10) & (frame.point_index == 3), "quality_flags"].iloc[0] == 4


def test_read_trajectories_returns_complete_histories(data):
    through = data.read_trajectories(start="2023-08-03", stop="2023-08-05")
    assert set(through.trajectory_id) == {10} and through.point_index.tolist() == [0, 1, 2, 3]
    boxed = data.read_trajectories(xlim=(290, 310), ylim=(40, 60))
    assert set(boxed.trajectory_id) == {10} and len(boxed) == 4
    both = data.read_trajectories(start="2024-01-02", xlim=(0, 30), ylim=(0, 10))
    assert set(both.trajectory_id) == {12} and both.point_index.tolist() == [0, 1]
    only = data.read_trajectories(trajectory_ids=[12], columns=["trajectory_id", "point_index", "x_m"])
    assert list(only) == ["trajectory_id", "point_index", "x_m"] and only.x_m.tolist() == [5.0, 25.0]
    assert data.read_trajectories(start="2030-01-01").empty
    with pytest.raises(ValueError, match="Give start/stop"):
        data.read_trajectories()


def test_read_displacements_pairs_consecutive_observations(data):
    frame = data.read_displacements(columns="all")
    assert len(frame) == 6 and frame["start_time"].is_monotonic_increasing
    row = frame[(frame.trajectory_id == 10) & (frame.point_index == 2)].iloc[0]
    assert (row.x0_m, row.y0_m, row.x1_m, row.y1_m) == (100.0, 0.0, 200.0, 50.0)
    assert row.start_time == pd.Timestamp("2023-07-31T12:00:00", tz="UTC")
    assert row.end_time == pd.Timestamp("2023-08-02", tz="UTC")
    assert (row.start_scene, row.end_scene) == (scene(2), scene(3))
    assert row["quality_flags"] == 2 and bool(row.interpolated) is True and not row.is_primary
    assert frame.attrs["period"] == PERIOD
    assert str(frame["start_time"].dtype) == "datetime64[ns, UTC]"
    assert row["elapsed_seconds"] == 129_600.0 and frame.attrs["excluded_flags"] == []
    # the window is on the displacement's start time: the one that starts before it is excluded
    window = data.read_displacements(start="2023-08-01", stop="2023-08-05")
    assert window[["trajectory_id", "point_index"]].values.tolist() == [[10, 3]]
    assert len(data.read_displacements(primary_only=True)) == 4
    assert len(data.read_displacements(trajectory_ids=[11])) == 2


def test_read_displacements_can_drop_flagged_displacements(data):
    false_land = data.read_displacements(exclude_flags="MASK_FALSE_LAND", columns="all")
    assert len(false_land) == 5 and (4 not in false_land["quality_flags"].tolist())
    assert false_land.attrs["excluded_flags"] == ["MASK_FALSE_LAND"]
    both = data.read_displacements(exclude_flags=["LOW_PAIR_DIRECT_SHARE", "MASK_FALSE_LAND"], columns=["quality_flags"])
    assert both["quality_flags"].tolist() == [0, 1, 0, 0]  # only FEW_NEIGHBOURS (bit 0) is left
    assert len(data.read_displacements(start="2023-08-01", stop="2023-08-05", exclude_flags="MASK_FALSE_LAND")) == 0
    with pytest.raises(ValueError, match="Unknown flags"):
        data.read_displacements(exclude_flags=["PAIR_INCOHERENT"])
    columns = data.read_displacements(columns=["trajectory_id", "speed_m_per_day"])
    assert list(columns) == ["trajectory_id", "speed_m_per_day"]
    with pytest.raises(ValueError, match="Unavailable"):
        data.read_displacements(columns=["distance_m"])


def test_unsupported_or_incomplete_files_are_rejected(tmp_path):
    future = tmp_path / "future.nc"
    write_netcdf(future, version="3.0.0")
    with pytest.raises(ValueError, match="Unsupported LiMOSAT schema"):
        open_limosat(future)
    other = tmp_path / "other.nc"
    netCDF4.Dataset(other, "w").close()
    with pytest.raises(ValueError, match="not a LiMOSAT trajectory file"):
        open_limosat(other)


def test_verify_checks_the_manifest_checksum(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = path.with_name("manifest.json")
    manifest.write_text(json.dumps({"files": [{"file": path.name, "sha256": digest}]}))
    with open_limosat(path) as data:
        assert data.verify() is True
    manifest.write_text(json.dumps({"files": [{"file": path.name, "sha256": "0" * 64}]}))
    with pytest.raises(ValueError, match="checksum differs"):
        open_limosat(path).verify()
    manifest.write_text(json.dumps({"files": []}))
    with pytest.raises(ValueError, match="not listed"):
        open_limosat(path).verify()
    manifest.unlink()
    with pytest.raises(FileNotFoundError):
        open_limosat(path).verify()


def test_open_rejects_everything_that_is_not_a_trajectory_file(tmp_path, path):
    with pytest.raises(FileNotFoundError):
        open_limosat(tmp_path / "missing.nc")
    with pytest.raises(ValueError, match="one period file"):
        open_limosat(tmp_path)
    feather = tmp_path / "benchmark.feather"
    feather.write_bytes(b"ARROW1")
    parquet = tmp_path / "limosat_old.parquet"
    parquet.write_bytes(b"PAR1")
    database = tmp_path / "raw.sqlite"
    sqlite3.connect(database).close()
    for other in (feather, parquet, database):
        with pytest.raises(ValueError, match="not a LiMOSAT trajectory file"):
            open_limosat(other)
    assert isinstance(open_limosat(path), LimosatTrajectories)
