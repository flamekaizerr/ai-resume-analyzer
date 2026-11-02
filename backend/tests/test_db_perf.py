import importlib.util
import math
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

backend_dir = str(Path(__file__).resolve().parents[1])
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from database import AnalysisRecord, Base, ensure_indexes
from db_queries import get_history, get_stats

repo_root = Path(__file__).resolve().parents[2]
seed_path = repo_root / "scripts" / "seed_db.py"
spec = importlib.util.spec_from_file_location("seed_db", seed_path)
seed_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed_db)


def _engine(path):
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    return engine


def _record(record_id, timestamp, score=0.5, matched=None, missing=None):
    return AnalysisRecord(
        id=record_id,
        timestamp=timestamp,
        similarity_score=score,
        matched_skills=matched or [],
        missing_skills=missing or [],
        transferable_skills=["Python -> FastAPI"],
        section_scores={"overall": score, "technical_skills": score},
        feedback=["Add measurable outcomes."],
    )


def test_seed_guard_and_bulk_seed(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="SPACE_ID"):
        seed_db.validate_seed_target(tmp_path / "blocked.db", {"SPACE_ID": "space"})
    with pytest.raises(ValueError, match="/data"):
        seed_db.validate_seed_target("/data/analyses.db", {})
    with pytest.raises(ValueError, match="DATABASE_URL"):
        seed_db.validate_seed_target(
            tmp_path / "live.db",
            {"DATABASE_URL": f"sqlite:////{str(tmp_path / 'live.db').lstrip('/')}"},
        )
    assert seed_db.validate_seed_target(tmp_path / "ok.db", {}) == (tmp_path / "ok.db").resolve()

    database_path = tmp_path / "seeded.db"
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = subprocess.run(
        [sys.executable, str(seed_path), "--db", str(database_path), "--rows", "500", "--batch", "73", "--seed", "42", "--days", "20"],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key not in {"DATABASE_URL", "SPACE_ID"}},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "500 analyses" in result.stdout
    engine = create_engine(f"sqlite:///{database_path}")
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM analyses")) == 500
    assert sorted(path.name for path in tmp_path.iterdir()) == ["seeded.db"]
    engine.dispose()


def test_ensure_indexes_migrates_old_schema_idempotently(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE analyses (id VARCHAR PRIMARY KEY, timestamp DATETIME, "
            "similarity_score FLOAT, matched_skills JSON, missing_skills JSON, "
            "transferable_skills JSON, section_scores JSON, feedback JSON)"
        ))
        connection.execute(text("CREATE INDEX ix_analyses_id ON analyses (id)"))
    ensure_indexes(engine)
    ensure_indexes(engine)
    with engine.connect() as connection:
        names = set(connection.scalars(text(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'analyses'"
        )))
    assert "ix_analyses_timestamp_id" in names
    assert "ix_analyses_id" not in names
    engine.dispose()


def test_history_pagination_and_filters(tmp_path):
    engine = _engine(tmp_path / "history.db")
    base = datetime(2025, 1, 1, 12)
    with Session(engine) as session:
        records = []
        for index in range(57):
            timestamp = base if index < 8 else base - timedelta(seconds=index)
            records.append(_record(
                f"id-{index:03d}", timestamp, score=index / 100,
                matched=["Python"], missing=["AWS"],
            ))
        session.add_all(records)
        session.commit()

        legacy = get_history(session)
        assert isinstance(legacy, list)
        assert len(legacy) == 57
        assert set(legacy[0]) == {
            "id", "timestamp", "similarity_score", "overall_score", "matched_skills", "missing_skills"
        }
        expected = sorted(records, key=lambda row: (row.timestamp, row.id), reverse=True)
        assert [item["id"] for item in legacy] == [row.id for row in expected]

        all_items = []
        cursor = None
        while True:
            page = get_history(session, limit=7, before=cursor)
            all_items.extend(page["items"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert [item["id"] for item in all_items] == [item["id"] for item in legacy]
        assert len({item["id"] for item in all_items}) == 57

        assert get_history(session, limit=10, before=base.isoformat())["items"]
        filtered = get_history(session, limit=100, min_score=0.5)
        assert all(item["similarity_score"] >= 0.5 for item in filtered["items"])
        assert len(filtered["items"]) == 7
        with pytest.raises(Exception) as error:
            get_history(session, limit=10, before="not-a-cursor")
        assert getattr(error.value, "status_code", None) == 400
    engine.dispose()


def _old_stats(session):
    records = session.query(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc()).all()
    if not records:
        return {
            "avg_score": 0.0,
            "best_score": 0.0,
            "recent_avg": 0.0,
            "total_analyses": 0,
            "top_missing_skills": [],
            "top_matched_skills": [],
            "last_analysis_at": None,
        }
    scores = [record.similarity_score for record in records]
    average = sum(scores) / len(scores)
    best = max(scores)
    recent = sum(scores[:5]) / len(scores[:5])
    missing_counts = {}
    matched_counts = {}
    for record in records:
        for skill in record.missing_skills or []:
            missing_counts[skill] = missing_counts.get(skill, 0) + 1
        for skill in record.matched_skills or []:
            matched_counts[skill] = matched_counts.get(skill, 0) + 1
    top_missing = sorted(missing_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    top_matched = sorted(matched_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "avg_score": average,
        "best_score": best,
        "recent_avg": recent,
        "total_analyses": len(records),
        "top_missing_skills": [{"skill": key, "count": count} for key, count in top_missing],
        "top_matched_skills": [{"skill": key, "count": count} for key, count in top_matched],
        "last_analysis_at": records[0].timestamp.isoformat() if records[0].timestamp else None,
    }


def _assert_stats_equal(actual, expected):
    assert math.isclose(actual["avg_score"], expected["avg_score"], rel_tol=1e-12)
    for key in expected.keys() - {"avg_score"}:
        assert actual[key] == expected[key]


def test_stats_match_old_implementation_on_seeded_data(tmp_path, monkeypatch):
    database_path = tmp_path / "stats_seed.db"
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    seed_db.seed_bulk(database_path, 3000, batch=500, seed=9, days=30)
    engine = create_engine(f"sqlite:///{database_path}")
    with Session(engine) as session:
        expected = _old_stats(session)
        actual = get_stats(session)
    _assert_stats_equal(actual, expected)
    engine.dispose()


def test_stats_tie_order_and_empty_table(tmp_path):
    engine = _engine(tmp_path / "ties.db")
    with Session(engine) as session:
        assert get_stats(session) == {
            "avg_score": 0.0,
            "best_score": 0.0,
            "recent_avg": 0.0,
            "total_analyses": 0,
            "top_missing_skills": [],
            "top_matched_skills": [],
            "last_analysis_at": None,
        }
        start = datetime(2025, 1, 1)
        session.add_all([
            _record("a", start, 0.2, ["Ruby", "Go"], ["Rust", "Elm"]),
            _record("b", start - timedelta(seconds=1), 0.4, ["Go", "Ruby"], ["Elm", "Rust"]),
            _record("c", start - timedelta(seconds=2), 0.6, ["TypeScript"], ["Scala"]),
        ])
        session.commit()
        stats = get_stats(session)
        assert stats["top_missing_skills"][:2] == [
            {"skill": "Rust", "count": 2}, {"skill": "Elm", "count": 2}
        ]
        assert stats["top_matched_skills"][:2] == [
            {"skill": "Ruby", "count": 2}, {"skill": "Go", "count": 2}
        ]
        _assert_stats_equal(stats, _old_stats(session))
    engine.dispose()


def test_stats_skip_null_skill_lists(tmp_path):
    engine = _engine(tmp_path / "nulls.db")
    with Session(engine) as session:
        record = _record("n", datetime(2025, 1, 1), 0.3)
        record.matched_skills = None
        record.missing_skills = None
        session.add_all([record, _record("p", datetime(2025, 1, 2), 0.7, ["Go"], ["AWS"])])
        session.commit()
        stats = get_stats(session)
        assert stats["top_matched_skills"] == [{"skill": "Go", "count": 1}]
        assert stats["top_missing_skills"] == [{"skill": "AWS", "count": 1}]
        _assert_stats_equal(stats, _old_stats(session))
    engine.dispose()


def test_seed_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    as_of = datetime(2026, 1, 1)
    for name in ("a.db", "b.db"):
        seed_db.seed_bulk(tmp_path / name, 200, batch=64, seed=5, as_of=as_of)
    rows = []
    for name in ("a.db", "b.db"):
        engine = create_engine(f"sqlite:///{tmp_path / name}")
        with engine.connect() as connection:
            rows.append(connection.execute(text(
                "SELECT id, timestamp, similarity_score, matched_skills FROM analyses ORDER BY id"
            )).all())
        engine.dispose()
    assert rows[0] == rows[1]


@pytest.fixture
def api_client(tmp_path):
    import httpx

    # same starlette/httpx workaround as test_agent.py
    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs.pop("app", None)
        return original(self, *args, **kwargs)

    httpx.Client.__init__ = patched
    from starlette.testclient import TestClient

    import main
    from database import get_db

    engine = _engine(tmp_path / "api.db")
    with Session(engine) as session:
        base = datetime(2025, 3, 1)
        session.add_all([
            _record(f"r{index:02d}", base - timedelta(minutes=index), index / 30, ["Python"], ["AWS"])
            for index in range(30)
        ])
        session.commit()

    def override():
        with Session(engine) as session:
            yield session

    main.app.dependency_overrides[get_db] = override
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.pop(get_db, None)
        httpx.Client.__init__ = original
        engine.dispose()


def test_history_endpoint(api_client):
    legacy = api_client.get("/history")
    assert legacy.status_code == 200
    assert isinstance(legacy.json(), list) and len(legacy.json()) == 30

    first = api_client.get("/history", params={"limit": 20}).json()
    assert len(first["items"]) == 20 and first["next_cursor"]
    second = api_client.get("/history", params={"limit": 20, "before": first["next_cursor"]}).json()
    assert len(second["items"]) == 10 and second["next_cursor"] is None
    ids = [item["id"] for item in first["items"] + second["items"]]
    assert ids == [item["id"] for item in legacy.json()]

    filtered = api_client.get("/history", params={"min_score": 0.9}).json()
    assert [item["id"] for item in filtered["items"]] == ["r27", "r28", "r29"]

    assert api_client.get("/history", params={"limit": 0}).status_code == 422
    assert api_client.get("/history", params={"limit": 101}).status_code == 422
    assert api_client.get("/history", params={"limit": 5, "before": "nope|x"}).status_code == 400


def test_stats_endpoint(api_client):
    body = api_client.get("/stats").json()
    assert body["total_analyses"] == 30
    assert body["top_matched_skills"] == [{"skill": "Python", "count": 30}]
    assert body["last_analysis_at"] == "2025-03-01T00:00:00"
