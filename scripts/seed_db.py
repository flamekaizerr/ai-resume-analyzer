import argparse
import json
import os
import random
import re
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy.engine import make_url


def _database_url_path(database_url):
    try:
        url = make_url(database_url)
    except (TypeError, ValueError):
        return None
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return None
    return Path(url.database).expanduser().resolve()


def validate_seed_target(db_path, environ=None):
    env = os.environ if environ is None else environ
    if db_path is None:
        raise ValueError("Bulk seeding requires --db with a local file path")
    if "SPACE_ID" in env:
        raise ValueError("Bulk seeding is disabled when SPACE_ID is set")
    if (
        "://" in str(db_path)
        or str(db_path).startswith(("sqlite:", "file:"))
        or str(db_path) == ":memory:"
        or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", str(db_path))
    ):
        raise ValueError("--db must be a plain local file path, not a database URL")

    resolved = Path(db_path).expanduser().resolve()
    data_dir = Path("/data").resolve()
    if resolved == data_dir or data_dir in resolved.parents:
        raise ValueError("Bulk seeding is disabled for paths under /data")

    configured_url = env.get("DATABASE_URL")
    configured_path = _database_url_path(configured_url) if configured_url else None
    if configured_path is not None and resolved == configured_path:
        raise ValueError("--db must not point to the file in DATABASE_URL")
    return resolved


def _load_model_module(database_path):
    backend_dir = str(Path(__file__).resolve().parent.parent / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    os.environ["DATABASE_URL"] = f"sqlite:///{database_path}"
    import database
    return database


def _skills():
    taxonomy_path = Path(__file__).resolve().parent.parent / "backend" / "data" / "skill_taxonomy.json"
    with taxonomy_path.open(encoding="utf-8") as taxonomy_file:
        values = json.load(taxonomy_file).values()
    return sorted(set(values))


def _record(rng, skills, now, days):
    score = rng.uniform(0.35, 0.98)
    matched = rng.sample(skills, rng.randint(0, min(8, len(skills))))
    missing = rng.sample(skills, rng.randint(0, min(8, len(skills))))
    timestamp = now - timedelta(
        microseconds=rng.randrange(days * 24 * 60 * 60 * 1_000_000 + 1)
    )
    return {
        # ids come from rng too, so the same --seed gives the same rows
        "id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
        "timestamp": timestamp,
        "similarity_score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "transferable_skills": [
            f"{source} -> {target}"
            for source, target in [
                rng.sample(skills, 2)
                for _ in range(rng.randint(0, min(3, len(skills) // 2)))
            ]
        ],
        "section_scores": {
            "technical_skills": rng.uniform(0.2, 1.0),
            "soft_skills": rng.uniform(0.3, 1.0),
            "experience_match": rng.uniform(0.2, 1.0),
            "education": rng.uniform(0.3, 1.0),
            "keywords": rng.uniform(0.2, 1.0),
            "overall": score,
        },
        "feedback": rng.sample([
            "Add more metrics to your experience section.",
            "Highlight projects that show relevant technical skills.",
            "Use keywords from the job description where they fit.",
            "Make your impact clear with specific outcomes.",
            "Add details about collaboration and communication.",
        ], rng.randint(1, 3)),
    }


def seed_bulk(db_path, rows, batch=5000, seed=42, days=365, create_tables=True, as_of=None):
    target = validate_seed_target(db_path)
    database = _load_model_module(target)
    from sqlalchemy import create_engine, insert

    engine = create_engine(f"sqlite:///{target}")
    if create_tables:
        database.Base.metadata.create_all(bind=engine)
    rng = random.Random(seed)
    skills = _skills()
    now = as_of or datetime.utcnow()
    statement = insert(database.AnalysisRecord.__table__)
    inserted = 0
    try:
        while inserted < rows:
            count = min(batch, rows - inserted)
            records = [_record(rng, skills, now, days) for _ in range(count)]
            with engine.begin() as connection:
                connection.execute(statement, records)
            inserted += count
    finally:
        engine.dispose()
    return inserted


def seed_legacy():
    backend_dir = str(Path(__file__).resolve().parent.parent / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from database import Base, SessionLocal, engine, AnalysisRecord

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for _ in range(10):
            score = random.uniform(0.4, 0.95)
            db.add(AnalysisRecord(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                similarity_score=score,
                matched_skills=["Python", "React", "Docker"],
                missing_skills=["Kubernetes", "AWS"] if score < 0.7 else [],
                transferable_skills=["JavaScript -> React"],
                section_scores={
                    "technical_skills": score,
                    "soft_skills": random.uniform(0.6, 0.9),
                    "experience_match": score - 0.1,
                    "education": 0.9,
                    "keywords": score,
                    "overall": score,
                },
                feedback=["Add more metrics to your experience section."],
            ))
        db.commit()
    finally:
        db.close()
    print("Database seeded with 10 past analyses.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    if args.rows is None:
        if args.db is not None:
            parser.error("--db is only available with --rows")
        seed_legacy()
        return
    if args.rows < 0 or args.batch < 1 or args.days < 1:
        parser.error("--rows must be non-negative, --batch and --days must be positive")
    try:
        inserted = seed_bulk(args.db, args.rows, args.batch, args.seed, args.days)
    except ValueError as error:
        parser.error(str(error))
    print(f"Database seeded with {inserted} analyses in {args.db}.")


if __name__ == "__main__":
    main()
