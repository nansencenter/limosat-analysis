import json
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

from limosat_analysis import open_limosat
from limosat_analysis.reader import QC_FLAG_BITS, QC_PROTOCOLS

WINDOW = "202305_202407"
PROTOCOL, PROTOCOL_SHA256 = next(iter(QC_PROTOCOLS.items()))
BIG_ID = 2**53 + 3

# Raw observations: (rowid, image_id, time, x, y, corr, interpolated, QC
# trajectory, QC is_last, original trajectory). Original trajectory 10 is split
# by a removed link into QC trajectories 10 and 11; the only link of 13 is
# removed, leaving single-point trajectories 13 and 14. Raw singleton 12, in no
# link, occurs in the 2015-22 periods and has no original ID.
POINTS = [
    (1, BIG_ID, "2023-07-29 00:00:00", 0, 0, .9, 0, 10, 0, 10),
    (2, BIG_ID + 1, "2023-07-30 00:00:00", 3, 4, .8, 1, 10, 0, 10),
    (3, BIG_ID + 2, "2023-07-31 00:00:00", 6, 8, .7, 0, 10, 1, 10),
    (4, 20, "2023-08-01 00:00:00", 100, 100, .6, 0, 11, 0, 10),
    (5, 21, "2023-08-02 00:00:00", 103, 104, .5, 0, 11, 1, 10),
    (6, 30, "2023-07-30 00:00:00", 1, 1, None, 0, 12, 1, 12),
    (7, 40, "2023-08-01 00:00:00", 500, 500, .4, 0, 13, 1, 13),
    (8, 41, "2023-08-02 00:00:00", 900, 500, .3, 1, 14, 1, 13),
]
# (source rowid, target rowid, link index, flags or removal reason)
KEPT = [(1, 2, 0, 0), (2, 3, 1, 2), (4, 5, 3, 1)]
REMOVED = [(3, 4, 2, "SPEED"), (7, 8, 0, "OPEN_WATER")]


def link_frame(rows):
    points = {row[0]: row for row in POINTS}
    # Link endpoints carry the original (pre-QC) is_last, unlike the QC SQLite.
    original_last = {max((row for row in POINTS if row[9] == original), key=lambda row: row[2])[0] for original in {row[9] for row in POINTS}}
    records = []
    for source, target, index, _ in rows:
        s, t = points[source], points[target]
        start, end = pd.Timestamp(s[2], tz="UTC"), pd.Timestamp(t[2], tz="UTC")
        dx, dy = t[3] - s[3], t[4] - s[4]
        elapsed = (end - start).total_seconds()
        record = {
            "edge_uid": f"edge-{source}-{target}", "original_trajectory_id": s[9], "trajectory_link_index": index,
            "source_rowid": source, "target_rowid": target, "source_image_id": s[1], "target_image_id": t[1],
            "start_time_utc": start, "end_time_utc": end, "elapsed_seconds": elapsed,
            "x0_m": float(s[3]), "y0_m": float(s[4]), "x1_m": float(t[3]), "y1_m": float(t[4]),
            "dx_m": float(dx), "dy_m": float(dy), "distance_m": float(np.hypot(dx, dy)),
            "speed_m_per_day": np.hypot(dx, dy) / elapsed * 86_400,
        }
        for prefix, point in (("source", s), ("target", t)):
            record.update({
                f"{prefix}_interpolated": point[6], f"{prefix}_corr": point[5], f"{prefix}_orbit_num": 1000 + point[0],
                f"{prefix}_is_last": int(point[0] in original_last), f"{prefix}_stopped": 0, f"{prefix}_converged_to": None,
            })
        record["pair_direct_share"] = .3 if index == 1 else .8
        records.append(record)
    frame = pd.DataFrame(records)
    for name in ("source_converged_to", "target_converged_to"):
        frame[name] = frame[name].astype("Int64")
    for name in ("source_interpolated", "target_interpolated", "source_is_last", "target_is_last", "source_stopped", "target_stopped"):
        frame[name] = frame[name].astype("int8")
    for name in ("start_time_utc", "end_time_utc"):
        frame[name] = frame[name].astype("datetime64[ms, UTC]")
    return frame


def kept_frame():
    frame = link_frame(KEPT)
    points = {row[0]: row for row in POINTS}
    frame["qc_trajectory_id"] = [points[source][7] for source, *_ in KEPT]
    frame["qc_segment"] = [0, 0, 1]
    frame["qc_flags"] = np.array([flags for *_, flags in KEPT], dtype="uint16")
    frame["qc_flag_reasons"] = ["|".join(name for name, bit in QC_FLAG_BITS.items() if flags >> bit & 1) for *_, flags in KEPT]
    frame["qc_sic_start"] = frame["qc_sic_end"] = np.float32(.9)
    frame["qc_neighbours_30km"] = np.array([5, 5, 1], dtype="int16")
    return frame


def removed_frame():
    frame = link_frame(REMOVED)
    frame["qc_exclusion"] = [reason for *_, reason in REMOVED]
    frame["sic_start"] = [.9, .05]
    frame["direct"] = [True, False]
    return frame


def write_kept(path, frame, protocol_id=PROTOCOL):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({b"limosat.qc": json.dumps({"protocol_id": protocol_id, "flag_bits": QC_FLAG_BITS}).encode()})
    pq.write_table(table, path)


def manifest(**changes):
    return {
        "status": "complete", "period": WINDOW, "protocol_id": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
        "input_links": 5, "kept_links": 3, "removed_links": 2, "continuity_violations": 0,
        "removed_by_step": {"SPEED": 1, "OPEN_WATER": 1}, "kept_links_by_flag": {"FEW_NEIGHBOURS": 1, "LOW_PAIR_DIRECT_SHARE": 1},
        **changes,
    }


@pytest.fixture
def release(tmp_path):
    folder = tmp_path / WINDOW
    folder.mkdir()
    write_kept(folder / f"{WINDOW}_qc.parquet", kept_frame())
    pq.write_table(pa.Table.from_pandas(removed_frame(), preserve_index=False), folder / f"{WINDOW}_qc_removed.parquet")
    (folder / f"{WINDOW}_qc.manifest.json").write_text(json.dumps(manifest(outputs={})))
    with sqlite3.connect(folder / f"{WINDOW}_qc.sqlite") as conn:
        conn.executescript(f'''
            CREATE TABLE "{WINDOW}__qc" (
                image_id BIGINT, is_last BIGINT, trajectory_id BIGINT, geometry TEXT,
                descriptors TEXT, angle FLOAT, corr FLOAT, time DATETIME,
                interpolated BIGINT, orbit_num BIGINT, stopped BOOLEAN, converged_to BIGINT
            );
            CREATE TABLE qc_metadata (key TEXT PRIMARY KEY, value_json TEXT);
        ''')
        # Deliberately unordered insertion with explicit rowids.
        conn.executemany(f'INSERT INTO "{WINDOW}__qc" (rowid, image_id, is_last, trajectory_id, geometry, descriptors, corr, time, interpolated, orbit_num, stopped) VALUES (?,?,?,?,?,?,?,?,?,?,?)', [
            (rowid, image, last, qc_id, f"POINT ({x} {y})", "excluded", corr, f"{time}.000000", interpolated, 1000 + rowid, 0)
            for rowid, image, time, x, y, corr, interpolated, qc_id, last, _ in reversed(POINTS)
        ])
        removed = removed_frame()
        for name in ("start_time_utc", "end_time_utc"):
            removed[name] = removed[name].astype(str)
        removed.to_sql("qc_removed_links", conn, index=False)
        conn.executemany("INSERT INTO qc_metadata VALUES (?, ?)", [(key, json.dumps(value)) for key, value in manifest().items()])
    return folder


@pytest.fixture
def qc(release):
    return release / f"{WINDOW}_qc.sqlite"


@pytest.fixture
def parquet(release):
    return release / f"{WINDOW}_qc.parquet"


def test_sqlite_native_points_and_qc_identity(qc):
    before = qc.read_bytes()
    data = open_limosat(qc)
    assert data.table == f"{WINDOW}__qc"
    assert data.metadata["qc_protocol_id"] == PROTOCOL and data.metadata["qc_protocol_sha256"] == PROTOCOL_SHA256
    assert data.metadata["period"] == WINDOW and data.metadata["crs"] == "EPSG:3413"
    points = data.read_points(include_singletons=True)
    assert len(points) == 8
    assert "including single-point" in points.attrs["point_coverage"]
    assert str(points.time_utc.dtype) == "datetime64[ns, UTC]"
    by_row = points.set_index("point_rowid")
    assert by_row.loc[[4, 5], "original_trajectory_id"].tolist() == [10, 10]
    # QC singleton 14 was split from original 13; raw singleton 12 is in no link.
    assert by_row.loc[8, "trajectory_id"] == 14 and by_row.loc[8, "original_trajectory_id"] == 13
    assert pd.isna(by_row.loc[6, "original_trajectory_id"])
    assert (points.qc_trajectory_id == points.trajectory_id).all()
    assert points.loc[points.trajectory_id == 10, "image_id"].tolist() == [BIG_ID, BIG_ID + 1, BIG_ID + 2]
    assert "descriptors" not in points and "geometry" not in points
    assert qc.read_bytes() == before


def test_singletons_excluded_by_default(qc):
    data = open_limosat(qc)
    points = data.read_points()
    assert sorted(points.point_rowid) == [1, 2, 3, 4, 5]
    assert points.attrs["point_coverage"].endswith("multi-point trajectories only")
    assert len(data.read_points(start="2023-08-01", stop="2023-08-02")) == 1
    selected, domain = data.read_trajectories_through(start="2023-08-01", stop="2023-08-03", xlim=(400, 1000), ylim=(400, 600))
    assert selected.empty and domain.empty
    selected, _ = data.read_trajectories_through(start="2023-08-01", stop="2023-08-03", xlim=(400, 1000), ylim=(400, 600), include_singletons=True)
    assert sorted(selected.trajectory_id) == [13, 14]


def test_original_ids_need_kept_links(qc, tmp_path):
    alone = tmp_path / "alone.sqlite"
    shutil.copy(qc, alone)
    data = open_limosat(alone)
    assert data.metadata["original_trajectory_ids"].startswith("unavailable")
    points = data.read_points(include_singletons=True)
    assert points.original_trajectory_id.isna().all()
    assert points.trajectory_id.notna().all()


def test_pair_before_date_filter_and_no_qc_bridge(qc):
    links = open_limosat(qc).read_displacements(start="2023-07-30", stop="2023-07-31")
    assert len(links) == 1
    edge = links.iloc[0]
    assert edge.source_image_id == BIG_ID + 1
    assert edge.target_image_id == BIG_ID + 2
    assert edge.end_time_utc == pd.Timestamp("2023-07-31", tz="UTC")
    assert edge.distance_m == 5 and edge.speed_m_per_day == 5
    assert "speed_m_s" not in links
    all_links = open_limosat(qc).read_displacements()
    assert len(all_links) == 3  # no link across the removed link or from singletons
    assert all_links.groupby("trajectory_id").size().to_dict() == {10: 2, 11: 1}
    assert (all_links.original_trajectory_id == 10).all()


def test_parquet_matches_sqlite_and_reconstructs_unique_endpoints(qc, parquet):
    sqlite_data, parquet_data = open_limosat(qc), open_limosat(parquet)
    columns = ["trajectory_id", "original_trajectory_id", "source_rowid", "target_rowid", "source_image_id", "target_image_id", "start_time_utc", "end_time_utc", "x0_m", "y0_m", "x1_m", "y1_m", "dx_m", "dy_m", "elapsed_seconds", "speed_m_per_day", "source_corr", "target_corr", "source_interpolated"]
    expected = sqlite_data.read_displacements(columns=columns)
    actual = parquet_data.read_displacements(columns=columns)
    assert_frame_equal(expected, actual, check_dtype=False)
    with pytest.raises(ValueError, match="no single-point"):
        parquet_data.read_points(include_singletons=True)
    points = parquet_data.read_points()
    assert len(points) == 5 and points.trajectory_id.nunique() == 2
    assert points.attrs["point_coverage"].startswith("kept-link endpoints")
    # By default both files give the same observations.
    shared = ["trajectory_id", "original_trajectory_id", "point_rowid", "image_id", "time_utc", "x_m", "y_m", "corr", "interpolated"]
    assert_frame_equal(sqlite_data.read_points(columns=shared), points[shared], check_dtype=False)
    assert points.image_id.max() == BIG_ID + 2
    assert len(parquet_data.read_points(start="2023-07-31", stop="2023-08-01")) == 1
    # The target-only date query must recover the incoming edge's endpoint.
    assert len(parquet_data.read_points(start="2023-08-02", stop="2023-08-03")) == 1


def test_parquet_exposes_qc_fields_and_manifest(parquet):
    data = open_limosat(parquet)
    assert data.kind == "parquet_qc_links"
    assert data.metadata["period"] == WINDOW and data.metadata["qc_protocol_sha256"] == PROTOCOL_SHA256
    assert data.metadata["ownership"] == {"policy_id": "limosat_release_august_annual_v1", "start": "2023-08-01", "stop_exclusive": "2024-08-01"}
    links = data.read_displacements()
    assert links.qc_flags.tolist() == [0, 2, 1]
    assert links.qc_flag_reasons.tolist() == ["", "LOW_PAIR_DIRECT_SHARE", "FEW_NEIGHBOURS"]
    # Trajectory 10 ends at a QC split; the original trajectory continues.
    assert links.target_is_last.tolist() == [0, 0, 1]
    assert {"edge_uid", "qc_segment", "pair_direct_share", "qc_sic_start", "qc_neighbours_30km", "source_stopped", "target_converged_to"} <= set(links)
    assert (links.period == WINDOW).all()
    assert str(links.start_time_utc.dtype) == "datetime64[ns, UTC]"


def test_exclude_flags_is_opt_in(parquet, qc):
    data = open_limosat(parquet)
    assert len(data.read_displacements()) == 3
    low = data.read_displacements(exclude_flags="LOW_PAIR_DIRECT_SHARE", columns=["trajectory_id", "qc_flags"])
    assert low.qc_flags.tolist() == [0, 1] and low.attrs["excluded_flags"] == ["LOW_PAIR_DIRECT_SHARE"]
    assert len(data.read_displacements(exclude_flags=["LOW_PAIR_DIRECT_SHARE", "FEW_NEIGHBOURS"], columns=["speed_m_per_day"])) == 1
    with pytest.raises(ValueError, match="Unknown QC flags"):
        data.read_displacements(exclude_flags=["PAIR_INCOHERENT"])
    with pytest.raises(ValueError, match="only in the QC v1 link Parquet"):
        open_limosat(qc).read_displacements(exclude_flags=["LOW_PAIR_DIRECT_SHARE"])


@pytest.mark.parametrize("source", ["qc", "parquet"])
def test_removed_links_with_reasons(request, source):
    data = open_limosat(request.getfixturevalue(source))
    removed = data.read_removed_links()
    assert removed.qc_exclusion.tolist() == ["SPEED", "OPEN_WATER"]
    assert removed.original_trajectory_id.tolist() == [10, 13]
    assert str(removed.start_time_utc.dtype) == "datetime64[ns, UTC]"
    assert removed.source_image_id.tolist() == [BIG_ID + 2, 40]
    assert removed.direct.tolist() == [True, False]
    owned = data.read_removed_links(owned_only=True, columns=["original_trajectory_id", "qc_exclusion"])
    assert owned.qc_exclusion.tolist() == ["OPEN_WATER"] and list(owned) == ["original_trajectory_id", "qc_exclusion"]
    assert data.read_removed_links(original_trajectory_ids=[10]).qc_exclusion.tolist() == ["SPEED"]
    assert data.read_removed_links(start="2023-07-31T00:00:00.000001Z").qc_exclusion.tolist() == ["OPEN_WATER"]
    assert data.read_removed_links(start="2023-07-30T23:59:59.5Z", stop="2023-07-31T00:00:00.000001Z").qc_exclusion.tolist() == ["SPEED"]


@pytest.mark.parametrize("source", ["qc", "parquet"])
def test_query_returns_full_histories_and_matching_indices(request, source):
    data = open_limosat(request.getfixturevalue(source))
    selected, domain = data.read_trajectories_through(start="2023-07-30", stop="2023-07-31", xlim=(2, 4), ylim=(3, 5))
    assert len(selected) == 3 and len(domain) == 1
    assert selected.time_utc.min() == pd.Timestamp("2023-07-29", tz="UTC")
    assert selected.time_utc.max() == pd.Timestamp("2023-07-31", tz="UTC")
    assert selected.loc[domain.index, "time_utc"].tolist() == domain.time_utc.tolist()
    empty, domain = data.read_trajectories_through(start="2023-07-30", stop="2023-07-31", xlim=(-5, -1), ylim=(-5, -1))
    assert empty.empty and domain.empty


@pytest.mark.parametrize("source", ["qc", "parquet"])
def test_release_ownership_is_half_open_on_start_time(request, source):
    data = open_limosat(request.getfixturevalue(source))
    owned = data.read_displacements(owned_only=True, columns=["trajectory_id", "start_time_utc"])
    assert owned.start_time_utc.tolist() == [pd.Timestamp("2023-08-01", tz="UTC")]
    assert owned.attrs["ownership_applied"]
    assert data.read_displacements(owned_only=True, start="2025-01-01").empty
    assert len(data.read_displacements(owned_only=True, stop="2023-08-01T00:00:00.001Z")) == 1


def test_projection_and_preview(parquet):
    data = open_limosat(parquet)
    columns = ["trajectory_id", "speed_m_per_day", "period"]
    projected = data.read_displacements(columns=columns)
    assert list(projected) == columns and len(projected) == 3
    assert len(data.read_displacements(max_rows=1)) == 1
    assert len(data.read_points(max_rows=1)) == 1
    with pytest.raises(ValueError, match="Unavailable"):
        data.read_displacements(columns=["speed_m_s"])
    with pytest.raises(ValueError, match="positive integer"):
        data.read_displacements(max_rows=0)


def test_column_projection_keeps_order_and_timezone_semantics(parquet):
    data = open_limosat(parquet)
    full = data.read_displacements(start="2023-07-30T02:00:00+02:00", stop="2023-08-02")
    projected = data.read_displacements(start="2023-07-30", stop="2023-08-02", columns=["speed_m_per_day"])
    assert full.speed_m_per_day.tolist() == projected.speed_m_per_day.tolist()
    assert len(full) == 2
    # A nanosecond after the source time must exclude that edge.
    assert len(data.read_displacements(start="2023-07-30T00:00:00.000000001Z", stop="2023-07-31")) == 0


@pytest.mark.parametrize(("change", "message"), [
    ({"status": "running"}, "complete"),
    ({"continuity_violations": 1}, "continuity"),
    ({"protocol_sha256": "0" * 64}, "Unsupported QC protocol"),
    ({"period": "202205_202307"}, "differs"),
])
def test_sqlite_qc_metadata_is_validated(qc, change, message):
    with sqlite3.connect(qc) as conn:
        for key, value in change.items():
            conn.execute("UPDATE qc_metadata SET value_json=? WHERE key=?", (json.dumps(value), key))
    with pytest.raises(ValueError, match=message):
        open_limosat(qc)


def test_parquet_manifest_and_schema_are_validated(release, parquet):
    sidecar = release / f"{WINDOW}_qc.manifest.json"
    sidecar.write_text(json.dumps(manifest(continuity_violations=2)))
    with pytest.raises(ValueError, match="continuity"):
        open_limosat(parquet)
    sidecar.unlink()
    with pytest.raises(ValueError, match="requires its manifest"):
        open_limosat(parquet)
    sidecar.write_text(json.dumps(manifest()))
    write_kept(parquet, kept_frame(), protocol_id="limosat_qc_v1_other")
    with pytest.raises(ValueError, match="disagrees"):
        open_limosat(parquet)
    write_kept(parquet, kept_frame().drop(columns="qc_flags"))
    with pytest.raises(ValueError, match="QC v1 link schema"):
        open_limosat(parquet)
    with pytest.raises(ValueError, match="removed-link file"):
        open_limosat(release / f"{WINDOW}_qc_removed.parquet")
    with pytest.raises(ValueError, match="one period"):
        open_limosat(release)


def test_qc_v2_sqlite_is_rejected(tmp_path):
    path = tmp_path / "v2.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(f'''
            CREATE TABLE "{WINDOW}__qc" (image_id INTEGER, trajectory_id INTEGER, time TEXT, geometry TEXT);
            CREATE TABLE qc_materialization_manifest(manifest_json TEXT);
        ''')
    with pytest.raises(ValueError, match="only LiMOSAT QC v1"):
        open_limosat(path)


def test_raw_sqlite_requires_crs(tmp_path):
    path = tmp_path / "raw.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(f'CREATE TABLE "{WINDOW}" (image_id INTEGER, trajectory_id INTEGER, time TEXT, geometry TEXT, is_last INTEGER)')
        conn.executemany(f'INSERT INTO "{WINDOW}" VALUES (?,?,?,?,?)', [
            (1, 5, "2023-05-01 00:00:00.000000", "POINT (0 0)", 0),
            (2, 5, "2023-05-02 00:00:00.000000", "POINT (3 4)", 1),
            (3, 6, "2023-05-01 00:00:00.000000", "POINT (9 9)", 1),
        ])
    with pytest.raises(ValueError, match="supply crs"):
        open_limosat(path)
    data = open_limosat(path, crs="EPSG:3413")
    assert data.metadata["qc_protocol_id"] is None and data.metadata["period"] == WINDOW
    points = data.read_points()
    assert points.trajectory_id.tolist() == [5, 5]
    assert len(data.read_points(include_singletons=True)) == 3
    assert "original_trajectory_id" not in points and "qc_trajectory_id" not in points
    assert data.read_displacements().speed_m_per_day.tolist() == [5.]
    with pytest.raises(ValueError, match="ownership"):
        data.read_displacements(owned_only=True)
    with pytest.raises(ValueError, match="QC v1 outputs"):
        data.read_removed_links()


def test_basic_feather_and_crs(tmp_path):
    path = tmp_path / "legacy.feather"
    feather.write_feather(pd.DataFrame({"g": [1, 1, 2], "d": pd.to_datetime(["2023-05-01", "2023-05-02", "2023-05-01"]), "x": [0., 3., 9.], "y": [0., 4., 9.]}), path)
    with pytest.raises(ValueError, match="supply crs"):
        open_limosat(path)
    data = open_limosat(path, crs="EPSG:3412")
    assert data.read_points().trajectory_id.tolist() == [1, 1]
    assert len(data.read_points(include_singletons=True, start="2023-05-01", stop="2023-05-02")) == 2
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
        conn.execute(f'UPDATE "{WINDOW}__qc" SET time=? WHERE trajectory_id=12', ("2023-07-30 00:00:00.000001",))
    assert len(data.read_points(start="2023-07-30T00:00:00.000001Z", stop="2023-07-30T00:00:00.000002Z", include_singletons=True)) == 1
    with sqlite3.connect(qc) as conn:
        conn.execute(f'UPDATE "{WINDOW}__qc" SET geometry="LINESTRING (0 0, 1 1)" WHERE trajectory_id=12')
    with pytest.raises(ValueError, match="POINT geometry"):
        data.read_points(include_singletons=True)


def test_repeated_time_retained_without_infinite_speed(qc):
    with sqlite3.connect(qc) as conn:
        conn.execute(f'UPDATE "{WINDOW}__qc" SET time="2023-07-29 00:00:00.000000" WHERE image_id=?', (BIG_ID + 1,))
    links = open_limosat(qc).read_displacements()
    zero = links.loc[links.elapsed_seconds == 0]
    assert len(zero) == 1
    assert zero.speed_m_per_day.isna().all() and not zero.positive_elapsed.any()
    assert not np.isinf(links.speed_m_per_day).any()


def test_shared_endpoint_conflict_is_reported(parquet):
    links = kept_frame()
    links.loc[links.source_rowid == 2, "x0_m"] += 1
    write_kept(parquet, links)
    with pytest.raises(ValueError, match="shared observation"):
        open_limosat(parquet).read_points()


def test_missing_point_time_is_reported(qc):
    with sqlite3.connect(qc) as conn:
        conn.execute(f'UPDATE "{WINDOW}__qc" SET time=NULL WHERE trajectory_id=12')
    with pytest.raises(ValueError, match="Missing observation timestamp"):
        open_limosat(qc).read_points(include_singletons=True)
