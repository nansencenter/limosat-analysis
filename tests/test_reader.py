import json
import sqlite3

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

from limosat_analysis import open_limosat

WINDOW = "202305_202407"
BIG_ID = 2**53 + 3


@pytest.fixture
def qc(tmp_path):
    path = tmp_path / "renamed.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(f'''
            CREATE TABLE "{WINDOW}__qc" (
                image_id INTEGER, trajectory_id INTEGER, time TEXT,
                geometry TEXT, corr REAL, interpolated INTEGER, is_last INTEGER,
                descriptors TEXT
            );
            CREATE INDEX trajectory_index ON "{WINDOW}__qc"(trajectory_id, is_last);
            CREATE TABLE qc_materialization_manifest(manifest_json TEXT);
            CREATE TABLE qc_break_assignment(new_trajectory_id INTEGER, original_trajectory_id INTEGER);
        ''')
        conn.execute("INSERT INTO qc_materialization_manifest VALUES (?)", (json.dumps({"status": "complete", "quick_check": "ok"}),))
        conn.execute("INSERT INTO qc_break_assignment VALUES (11, 10)")
        # Deliberately unordered insertion; trajectory 11 is the post-QC
        # continuation of original trajectory 10, while 12 is a singleton.
        conn.executemany(f'INSERT INTO "{WINDOW}__qc" VALUES (?,?,?,?,?,?,?,?)', [
            (BIG_ID + 1, 10, "2023-05-02 00:00:00.000000", "POINT (3 4)", .8, 1, 0, "excluded"),
            (BIG_ID, 10, "2023-05-01 00:00:00.000000", "POINT (0 0)", .9, 0, 0, "excluded"),
            (BIG_ID + 2, 10, "2023-05-03 00:00:00.000000", "POINT (6 8)", .7, 0, 1, "excluded"),
            (20, 11, "2023-05-04 00:00:00.000000", "POINT (100 100)", .6, 0, 0, "excluded"),
            (21, 11, "2023-05-05 00:00:00.000000", "POINT (103 104)", .5, 0, 1, "excluded"),
            (30, 12, "2023-05-02 00:00:00.000000", "POINT (1 1)", None, 0, 1, "excluded"),
        ])
    return path


def write_v2(path, links):
    out = links.drop(columns="trajectory_id").copy()
    out["qc_release_id"] = "example"
    out["edge_uid"] = [f"edge-{i}" for i in range(len(out))]
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), path)
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "window": WINDOW, "source_qc_release_id": "example",
        "canonical_ownership": {"start": "2023-05-02T00:00:00Z", "stop_exclusive": "2023-05-04T00:00:00Z"},
    }))
    return path


@pytest.fixture
def parquet(qc, tmp_path):
    return write_v2(tmp_path / f"{WINDOW}.parquet", open_limosat(qc).read_displacements())


def test_sqlite_native_points_and_qc_identity(qc):
    before = qc.read_bytes()
    data = open_limosat(qc)
    assert data.table == f"{WINDOW}__qc"
    points = data.read_points()
    assert len(points) == 6
    assert str(points.time_utc.dtype) == "datetime64[ns, UTC]"
    assert points.loc[points.trajectory_id == 11, "original_trajectory_id"].tolist() == [10, 10]
    assert points.loc[points.trajectory_id == 10, "image_id"].tolist() == [BIG_ID, BIG_ID + 1, BIG_ID + 2]
    assert "descriptors" not in points and "geometry" not in points
    assert set(["i", "g", "d", "x", "y", "q"]).isdisjoint(points.columns)
    assert qc.read_bytes() == before


def test_pair_before_date_filter_and_no_qc_bridge(qc):
    links = open_limosat(qc).read_displacements(start="2023-05-02", stop="2023-05-03")
    assert len(links) == 1
    edge = links.iloc[0]
    assert edge.source_image_id == BIG_ID + 1
    assert edge.target_image_id == BIG_ID + 2
    assert edge.end_time_utc == pd.Timestamp("2023-05-03", tz="UTC")
    assert edge.distance_m == 5 and edge.speed_m_per_day == 5
    all_links = open_limosat(qc).read_displacements()
    assert len(all_links) == 3  # points minus QC trajectories; no singleton edge
    assert all_links.groupby("trajectory_id").size().to_dict() == {10: 2, 11: 1}
    assert (all_links.source_is_last == 0).all()


def test_v2_matches_sqlite_and_reconstructs_unique_endpoints(qc, parquet):
    sqlite_data, parquet_data = open_limosat(qc), open_limosat(parquet)
    columns = ["trajectory_id", "source_image_id", "target_image_id", "start_time_utc", "end_time_utc", "x0_m", "y0_m", "x1_m", "y1_m", "dx_m", "dy_m", "elapsed_seconds", "speed_m_per_day", "source_corr", "target_corr"]
    assert_frame_equal(sqlite_data.read_displacements(columns=columns), parquet_data.read_displacements(columns=columns))
    with pytest.raises(ValueError, match="singleton"):
        parquet_data.read_points()
    points = parquet_data.read_points(linked_only=True)
    assert len(points) == 5 and points.trajectory_id.nunique() == 2
    assert points.attrs["point_coverage"].startswith("linked endpoints")
    assert len(parquet_data.read_points(linked_only=True, start="2023-05-03", stop="2023-05-04")) == 1
    # The target-only date query must recover the incoming edge's endpoint.
    assert len(parquet_data.read_points(linked_only=True, start="2023-05-05", stop="2023-05-06")) == 1


@pytest.mark.parametrize("source", ["qc", "parquet"])
def test_query_returns_full_histories_and_matching_indices(request, source):
    data = open_limosat(request.getfixturevalue(source))
    selected, domain = data.read_trajectories_through(start="2023-05-02", stop="2023-05-03", xlim=(2, 4), ylim=(3, 5), linked_only=True)
    assert len(selected) == 3 and len(domain) == 1
    assert selected.time_utc.min() == pd.Timestamp("2023-05-01", tz="UTC")
    assert selected.time_utc.max() == pd.Timestamp("2023-05-03", tz="UTC")
    assert selected.loc[domain.index, "time_utc"].tolist() == domain.time_utc.tolist()
    empty, domain = data.read_trajectories_through(start="2023-05-02", stop="2023-05-03", xlim=(-5, -1), ylim=(-5, -1), linked_only=True)
    assert empty.empty and domain.empty


def test_ownership_projection_and_preview(parquet):
    data = open_limosat(parquet)
    columns = ["trajectory_id", "speed_m_per_day", "canonical_source_window"]
    owned = data.read_displacements(owned_only=True, columns=columns)
    assert len(owned) == 1 and owned.attrs["ownership_applied"]
    assert list(owned) == columns
    assert data.read_displacements(owned_only=True, start="2024-01-01").empty
    assert len(data.read_displacements(max_rows=1)) == 1
    assert len(data.read_points(linked_only=True, max_rows=1)) == 1
    with pytest.raises(ValueError, match="Unavailable"):
        data.read_displacements(columns=["invented"])
    with pytest.raises(ValueError, match="positive integer"):
        data.read_displacements(max_rows=0)


def test_old_publisher_missing_diagnostics(qc, tmp_path):
    links = open_limosat(qc).read_displacements()
    cols = ["trajectory_id", "source_image_id", "target_image_id", "start_time_utc", "end_time_utc", "elapsed_seconds", "x0_m", "y0_m", "x1_m", "y1_m", "dx_m", "dy_m", "distance_m", "speed_m_s", "speed_m_per_day"]
    table = pa.Table.from_pandas(links[cols], preserve_index=False)
    table = table.append_column("source_position_type", pa.array(["observed", "interpolated", "observed"]))
    table = table.append_column("target_position_type", pa.array(["interpolated", "observed", "observed"]))
    table = table.replace_schema_metadata({b"limosat.crs": b"EPSG:3413", b"limosat.source_window": WINDOW.encode(), b"limosat.release_id": b"example"})
    path = tmp_path / "old.parquet"
    pq.write_table(table, path)
    data = open_limosat(path)
    out = data.read_displacements()
    assert out.source_corr.isna().all() and out.target_corr.isna().all()
    assert out.target_interpolated.tolist() == [1, 0, 0]
    assert "original_trajectory_id" not in out
    assert len(data.read_points(linked_only=True)) == 5


def test_monthly_store_and_package_root(tmp_path, qc):
    links = open_limosat(qc).read_displacements()
    store = tmp_path / "store"
    part = store / "partitions/year=2023/month=05/displacements.parquet"
    part.parent.mkdir(parents=True)
    write_v2(part, links)
    (store / "manifest.json").write_text(json.dumps({"status": "complete", "store_format_version": 2, "window": WINDOW, "partitions": [{"path": str(part.relative_to(store))}]}))
    assert len(open_limosat(store).read_displacements()) == 3
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text(json.dumps({"windows": {WINDOW: {}}}))
    with pytest.raises(ValueError, match="one period"):
        open_limosat(package)


def test_basic_feather_and_crs(tmp_path):
    path = tmp_path / "legacy.feather"
    feather.write_feather(pd.DataFrame({"g": [1, 1], "d": pd.to_datetime(["2023-05-01", "2023-05-02"]), "x": [0., 3.], "y": [0., 4.]}), path)
    with pytest.raises(ValueError, match="supply crs"):
        open_limosat(path)
    data = open_limosat(path, crs="EPSG:3412")
    assert data.read_points().image_id.isna().all()
    assert data.read_points()["corr"].isna().all()
    assert data.read_displacements().speed_m_per_day.tolist() == [5.]
    assert data.metadata["crs"] == "EPSG:3412"


def test_geometry_date_and_precision_errors(qc):
    data = open_limosat(qc)
    with pytest.raises(ValueError, match="start < stop"):
        data.read_points(start="2023-05-01", stop="2023-05-01")
    with pytest.raises(ValueError, match="does not reproject"):
        open_limosat(qc, crs="EPSG:3412")
    with sqlite3.connect(qc) as conn:
        conn.execute(f'UPDATE "{WINDOW}__qc" SET time=? WHERE trajectory_id=12', ("2023-05-02 00:00:00.000001",))
    assert len(data.read_points(start="2023-05-02T00:00:00.000001Z", stop="2023-05-02T00:00:00.000002Z")) == 1
    with sqlite3.connect(qc) as conn:
        conn.execute(f'UPDATE "{WINDOW}__qc" SET geometry="LINESTRING (0 0, 1 1)" WHERE trajectory_id=12')
    with pytest.raises(ValueError, match="POINT geometry"):
        data.read_points()


def test_repeated_time_retained_without_infinite_speed(qc):
    with sqlite3.connect(qc) as conn:
        conn.execute(f'UPDATE "{WINDOW}__qc" SET time="2023-05-01 00:00:00.000000" WHERE image_id=?', (BIG_ID + 1,))
    links = open_limosat(qc).read_displacements()
    zero = links.loc[links.elapsed_seconds == 0]
    assert len(zero) == 1
    assert zero.speed_m_s.isna().all() and not zero.positive_elapsed.any()
    assert not np.isinf(links.speed_m_s).any()


def test_column_projection_keeps_order_and_timezone_semantics(parquet):
    data = open_limosat(parquet)
    full = data.read_displacements(start="2023-05-02T02:00:00+02:00", stop="2023-05-05")
    projected = data.read_displacements(start="2023-05-02", stop="2023-05-05", columns=["speed_m_per_day"])
    assert full.speed_m_per_day.tolist() == projected.speed_m_per_day.tolist()
    assert len(full) == 2
    # A nanosecond after the source time must exclude that edge.
    assert len(data.read_displacements(start="2023-05-02T00:00:00.000000001Z", stop="2023-05-03")) == 0


def test_shared_endpoint_conflict_is_reported(qc, tmp_path):
    links = open_limosat(qc).read_displacements()
    links.loc[links.source_image_id == BIG_ID + 1, "x0_m"] += 1
    path = write_v2(tmp_path / "inconsistent.parquet", links)
    with pytest.raises(ValueError, match="shared observation"):
        open_limosat(path).read_points(linked_only=True)


def test_missing_point_time_is_reported(qc):
    with sqlite3.connect(qc) as conn:
        conn.execute(f'UPDATE "{WINDOW}__qc" SET time=NULL WHERE trajectory_id=12')
    with pytest.raises(ValueError, match="Missing observation timestamp"):
        open_limosat(qc).read_points()


def test_multiple_source_windows_cannot_share_unscoped_ids(qc, tmp_path):
    links = open_limosat(qc).read_displacements()
    links.loc[links.index[-1], "canonical_source_window"] = "other_period"
    path = write_v2(tmp_path / "mixed.parquet", links)
    with pytest.raises(ValueError, match="one source window"):
        open_limosat(path).read_displacements(columns=["trajectory_id", "speed_m_s"])
