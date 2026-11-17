import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from limosat_analysis import LimosatTrajectories, open_limosat

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
COLUMNS = ["trajectory_id", "point_index", "time", "scene", "x_m", "y_m", "dx_m", "dy_m", "speed_m_per_day",
           "interpolated", "correlation", "direct_match_share", "quality_flags", "is_primary"]


def write_points(path, version="2.0.0", rows=ROWS):
    kinds = {"period": pa.string(), "trajectory_id": pa.int64(), "point_index": pa.int32(), "time": pa.timestamp("ms", tz="UTC"),
             "scene": pa.string(), "interpolated": pa.bool_(), "correlation": pa.float32(),
             "direct_match_share": pa.float32(), "quality_flags": pa.uint8(), "is_primary": pa.bool_()}
    names = ["period"] + COLUMNS
    data = {name: [] for name in names}
    for row in rows:
        data["period"].append(PERIOD)
        for name, value in zip(COLUMNS, row):
            data[name].append(t(value) if name == "time" else scene(value) if name == "scene" else value)
    fields = [pa.field(name, kinds.get(name, pa.float64()),
                       metadata={b"unit": b"m" if name.endswith("_m") else b"", b"description": f"{name} text".encode()})
              for name in names]
    table = pa.table({name: pa.array(data[name], type=kinds.get(name, pa.float64())) for name in names},
                     schema=pa.schema(fields))
    table = table.replace_schema_metadata({
        b"limosat.schema_version": version.encode(), b"limosat.release_id": b"test-release",
        b"limosat.period": PERIOD.encode(), b"limosat.crs": b"EPSG:3413",
        b"limosat.flag_bits": json.dumps(FLAG_BITS).encode(),
        b"limosat.primary_window": json.dumps({"start": "2023-08-01T00:00:00Z", "stop_exclusive": "2024-08-01T00:00:00Z"}).encode(),
        b"limosat.primary_policy": b"limosat_release_august_annual_v1",
    })
    pq.write_table(table, path, row_group_size=3)


@pytest.fixture
def points(tmp_path):
    path = tmp_path / f"limosat_{PERIOD}.parquet"
    write_points(path)
    return path


def test_open_dispatches_to_the_trajectory_reader_and_reads_the_file_metadata(points):
    data = open_limosat(points)
    assert isinstance(data, LimosatTrajectories) and data.kind == "parquet_points"
    assert data.metadata["period"] == PERIOD and data.metadata["crs"] == "EPSG:3413"
    assert data.metadata["flag_bits"] == FLAG_BITS and data.metadata["rows"] == 9
    assert data.metadata["primary_window"]["stop_exclusive"] == "2024-08-01T00:00:00Z"
    description = data.describe().set_index("column")
    assert description.loc["x_m", "unit"] == "m" and description.loc["quality_flags", "description"] == "quality_flags text"


def test_read_points_filters_and_sorts_by_time(points):
    data = open_limosat(points)
    frame = data.read_points()
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


def test_read_trajectories_returns_complete_histories(points):
    data = open_limosat(points)
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


def test_read_displacements_pairs_consecutive_observations(points):
    data = open_limosat(points)
    frame = data.read_displacements()
    assert len(frame) == 6 and frame["start_time"].is_monotonic_increasing
    row = frame[(frame.trajectory_id == 10) & (frame.point_index == 2)].iloc[0]
    assert (row.x0_m, row.y0_m, row.x1_m, row.y1_m) == (100.0, 0.0, 200.0, 50.0)
    assert row.start_time == pd.Timestamp("2023-07-31T12:00:00", tz="UTC")
    assert row.end_time == pd.Timestamp("2023-08-02", tz="UTC")
    assert (row.start_scene, row.end_scene) == (scene(2), scene(3))
    assert row["quality_flags"] == 2 and bool(row.interpolated) is True and not row.is_primary and row.period == PERIOD
    assert str(frame["start_time"].dtype) == "datetime64[ns, UTC]"
    # the window is on the displacement's start time: the one that starts before it is excluded
    window = data.read_displacements(start="2023-08-01", stop="2023-08-05")
    assert window[["trajectory_id", "point_index"]].values.tolist() == [[10, 3]]
    assert len(data.read_displacements(primary_only=True)) == 4
    assert len(data.read_displacements(trajectory_ids=[11])) == 2


def test_read_displacements_can_drop_flagged_displacements(points):
    data = open_limosat(points)
    false_land = data.read_displacements(exclude_flags="MASK_FALSE_LAND")
    assert len(false_land) == 5 and (4 not in false_land["quality_flags"].tolist())
    assert false_land.attrs["excluded_flags"] == ["MASK_FALSE_LAND"]
    both = data.read_displacements(exclude_flags=["LOW_PAIR_DIRECT_SHARE", "MASK_FALSE_LAND"])
    assert both["quality_flags"].tolist() == [0, 1, 0, 0]  # only FEW_NEIGHBOURS (bit 0) is left
    assert len(data.read_displacements(start="2023-08-01", stop="2023-08-05", exclude_flags="MASK_FALSE_LAND")) == 0
    with pytest.raises(ValueError, match="Unknown flags"):
        data.read_displacements(exclude_flags=["PAIR_INCOHERENT"])
    columns = data.read_displacements(columns=["trajectory_id", "speed_m_per_day"])
    assert list(columns) == ["trajectory_id", "speed_m_per_day"]
    with pytest.raises(ValueError, match="Unavailable"):
        data.read_displacements(columns=["distance_m"])


def test_unsupported_or_incomplete_files_are_rejected(tmp_path):
    future = tmp_path / "future.parquet"
    write_points(future, version="3.0.0")
    with pytest.raises(ValueError, match="Unsupported LiMOSAT schema"):
        open_limosat(future)
    table = pq.read_table(tmp_path / "future.parquet").drop_columns(["x_m"])
    broken = tmp_path / "broken.parquet"
    pq.write_table(table.replace_schema_metadata({b"limosat.schema_version": b"2.0.0"}), broken)
    with pytest.raises(ValueError, match="lacks columns"):
        open_limosat(broken)


def test_verify_checks_the_manifest_checksum(points):
    digest = hashlib.sha256(points.read_bytes()).hexdigest()
    manifest = points.with_name("manifest.json")
    manifest.write_text(json.dumps({"files": [{"file": points.name, "sha256": digest}]}))
    assert open_limosat(points).verify() is True
    manifest.write_text(json.dumps({"files": [{"file": points.name, "sha256": "0" * 64}]}))
    with pytest.raises(ValueError, match="checksum differs"):
        open_limosat(points).verify()
    manifest.write_text(json.dumps({"files": []}))
    with pytest.raises(ValueError, match="not listed"):
        open_limosat(points).verify()
    manifest.unlink()
    with pytest.raises(FileNotFoundError):
        open_limosat(points).verify()
