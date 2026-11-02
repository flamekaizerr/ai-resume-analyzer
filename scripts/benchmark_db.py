import argparse
import json
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session


def old_history(session):
    records = session.query(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc()).all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "similarity_score": r.similarity_score,
            "overall_score": (r.section_scores or {}).get("overall", r.similarity_score),
            "matched_skills": r.matched_skills or [],
            "missing_skills": r.missing_skills or [],
        }
        for r in records
    ]


def old_stats(session):
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

    scores = [r.similarity_score for r in records]
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    recent_scores = scores[:5]
    recent_avg = sum(recent_scores) / len(recent_scores)

    missing_counts = {}
    matched_counts = {}
    for record in records:
        for skill in record.missing_skills or []:
            missing_counts[skill] = missing_counts.get(skill, 0) + 1
        for skill in record.matched_skills or []:
            matched_counts[skill] = matched_counts.get(skill, 0) + 1

    top_missing = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_matched = sorted(matched_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "avg_score": avg_score,
        "best_score": best_score,
        "recent_avg": recent_avg,
        "total_analyses": len(records),
        "top_missing_skills": [{"skill": k, "count": v} for k, v in top_missing],
        "top_matched_skills": [{"skill": k, "count": v} for k, v in top_matched],
        "last_analysis_at": records[0].timestamp.isoformat() if records[0].timestamp else None,
    }


def _capture_sql(engine, function):
    statements = []

    def collect(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", collect)
    try:
        with Session(engine) as session:
            function(session)
    finally:
        event.remove(engine, "before_cursor_execute", collect)
    explained = []
    with engine.connect() as connection:
        for statement, parameters in statements:
            try:
                plan = connection.exec_driver_sql(
                    "EXPLAIN QUERY PLAN " + statement, parameters
                ).all()
                explained.append({
                    "sql": statement,
                    "plan": [list(row) for row in plan],
                })
            except Exception as error:
                explained.append({"sql": statement, "error": str(error)})
    return explained


def _measure(engine, function, warmup, runs):
    for _ in range(warmup):
        with Session(engine) as session:
            function(session)
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        with Session(engine) as session:
            function(session)
        timings.append((time.perf_counter() - started) * 1000)
    ordered = sorted(timings)
    quantiles = statistics.quantiles(ordered, n=100)
    return {
        "raw_ms": timings,
        "summary": {
            "p50_ms": statistics.median(timings),
            "p95_ms": quantiles[94],
            "mean_ms": statistics.mean(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
        },
    }


def _cpu_model():
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or "unknown"
    match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    return match.group(1).strip() if match else platform.processor() or "unknown"


def _machine_info():
    import sqlalchemy
    import sqlite3

    return {
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "sqlalchemy_version": sqlalchemy.__version__,
        "date": datetime.now().astimezone().isoformat(),
        "note": "Benchmark ran on a shared VM.",
    }


def _summary_markdown(results):
    lines = [
        "# Database benchmark results",
        "",
        "Each call used a new SQLAlchemy Session. Percentiles use statistics.quantiles with n=100.",
        "",
        "| Rows | Case | Old p50 (ms) | New p50 (ms) | Old p95 (ms) | New p95 (ms) | p50 speedup | p95 speedup |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for size, cases in results.items():
        for case in ("history_first_page", "stats"):
            old = cases[f"old_{case}"]["summary"]
            new = cases[f"new_{case}"]["summary"]
            lines.append(
                f"| {size} | {case} | {old['p50_ms']:.3f} | {new['p50_ms']:.3f} | "
                f"{old['p95_ms']:.3f} | {new['p95_ms']:.3f} | "
                f"{old['p50_ms'] / new['p50_ms']:.2f} | {old['p95_ms'] / new['p95_ms']:.2f} |"
            )
        old_history_summary = cases["old_history_first_page"]["summary"]
        for case in ("new_history_deep_page", "new_history_full_list"):
            extra = cases[case]["summary"]
            lines.append(
                f"| {size} | {case} | {old_history_summary['p50_ms']:.3f} | {extra['p50_ms']:.3f} | "
                f"{old_history_summary['p95_ms']:.3f} | {extra['p95_ms']:.3f} | "
                f"{old_history_summary['p50_ms'] / extra['p50_ms']:.2f} | "
                f"{old_history_summary['p95_ms'] / extra['p95_ms']:.2f} |"
            )
    lines.extend([
        "",
        "Old history is the old handler, which always read every row. New history first page is limit=20 with no cursor.",
        "The deep page is limit=20 with a cursor near the middle of the table.",
        "new_history_full_list is the new code called with no params (the backward compatible full list), on the new schema.",
        "Timed: the handler function called with a new Session, DB query plus Python processing, time.perf_counter, no HTTP and no JSON serialisation.",
        "p50 is statistics.median and p95 is statistics.quantiles(n=100)[94] over the timed runs.",
    ])
    return "\n".join(lines) + "\n"


def run_benchmark(sizes, warmup, runs, out_dir, seed):
    global AnalysisRecord, query_history, query_stats, seed_bulk
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    explain_before = []
    explain_after = []

    with tempfile.TemporaryDirectory(prefix="db-benchmark-") as temp_dir:
        temp_root = Path(temp_dir)
        for size in sizes:
            old_path = temp_root / f"old_{size}.db"
            new_path = temp_root / f"new_{size}.db"
            seed_script = Path(__file__).resolve().parent / "seed_db.py"
            sys.path.insert(0, str(seed_script.parent))
            from seed_db import seed_bulk as seed_function
            seed_bulk = seed_function

            seed_bulk(
                old_path, size, batch=5000, seed=seed, create_tables=True,
                as_of=datetime(2026, 11, 2),
            )
            old_engine = create_engine(f"sqlite:///{old_path}")
            with old_engine.begin() as connection:
                connection.execute(text("DROP INDEX IF EXISTS ix_analyses_timestamp_id"))
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_analyses_id ON analyses (id)"
                ))
            seed_bulk(
                new_path, size, batch=5000, seed=seed, create_tables=True,
                as_of=datetime(2026, 11, 2),
            )
            new_engine = create_engine(f"sqlite:///{new_path}")

            if AnalysisRecord is None:
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
                from database import AnalysisRecord as model
                from db_queries import get_history, get_stats
                AnalysisRecord = model
                query_history = get_history
                query_stats = get_stats

            with new_engine.connect() as connection:
                oldest_middle = connection.execute(
                    text("SELECT timestamp, id FROM analyses ORDER BY timestamp DESC, id DESC LIMIT 1 OFFSET :offset"),
                    {"offset": size // 2},
                ).first()
            cursor = f"{oldest_middle.timestamp}|{oldest_middle.id}"
            old_history_fn = old_history
            old_stats_fn = old_stats
            new_history_fn = lambda session: query_history(session, limit=20)
            deep_history_fn = lambda session: query_history(session, limit=20, before=cursor)
            full_history_fn = lambda session: query_history(session)
            new_stats_fn = query_stats

            cases = {}
            # same rows in both DBs (same seed and as_of), only the schema and code differ
            for name, engine, function in [
                ("old_history_first_page", old_engine, old_history_fn),
                ("old_stats", old_engine, old_stats_fn),
                ("new_history_first_page", new_engine, new_history_fn),
                ("new_history_deep_page", new_engine, deep_history_fn),
                ("new_history_full_list", new_engine, full_history_fn),
                ("new_stats", new_engine, new_stats_fn),
            ]:
                print(f"{size} rows: timing {name}", flush=True)
                cases[name] = _measure(engine, function, warmup, runs)
                plans = _capture_sql(engine, function)
                line = f"Rows: {size}, case: {name}\n" + json.dumps(plans, indent=2)
                (explain_before if name.startswith("old_") else explain_after).append(line)
            results[str(size)] = cases
            old_engine.dispose()
            new_engine.dispose()

    output = {
        "method": "Each timed call used a new SQLAlchemy Session. p95 uses statistics.quantiles with n=100.",
        "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (output_dir / "machine.json").write_text(json.dumps(_machine_info(), indent=2), encoding="utf-8")
    (output_dir / "explain_before.txt").write_text("\n\n".join(explain_before) + "\n", encoding="utf-8")
    (output_dir / "explain_after.txt").write_text("\n\n".join(explain_after) + "\n", encoding="utf-8")
    summary = _summary_markdown(results)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[50000, 100000])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--out", default="scripts/benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(args.sizes, args.warmup, args.runs, args.out, args.seed)


AnalysisRecord = None
query_history = None
query_stats = None
seed_bulk = None


if __name__ == "__main__":
    main()
