from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import Integer, func, select, true, tuple_
from sqlalchemy.orm import Session

from database import AnalysisRecord


def _history_statement():
    table = AnalysisRecord.__table__
    overall_score = func.coalesce(
        func.json_extract(table.c.section_scores, "$.overall"),
        table.c.similarity_score,
    ).label("overall_score")
    return select(
        table.c.id,
        table.c.timestamp,
        table.c.similarity_score,
        overall_score,
        table.c.matched_skills,
        table.c.missing_skills,
    )


def _history_item(row):
    return {
        "id": row.id,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "similarity_score": row.similarity_score,
        "overall_score": row.overall_score,
        "matched_skills": row.matched_skills or [],
        "missing_skills": row.missing_skills or [],
    }


def _parse_cursor(cursor):
    if "|" in cursor:
        timestamp_text, record_id = cursor.rsplit("|", 1)
        if not record_id:
            raise ValueError
    else:
        timestamp_text, record_id = cursor, None
    try:
        timestamp = datetime.fromisoformat(timestamp_text)
    except (TypeError, ValueError):
        raise ValueError from None
    return timestamp, record_id


def get_history(session: Session, limit=None, before=None, min_score=None):
    paginated = limit is not None or before is not None or min_score is not None
    table = AnalysisRecord.__table__
    statement = _history_statement()

    if min_score is not None:
        statement = statement.where(table.c.similarity_score >= min_score)
    if before is not None:
        try:
            timestamp, record_id = _parse_cursor(before)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid before cursor") from None
        if record_id is None:
            statement = statement.where(table.c.timestamp < timestamp)
        else:
            # row value compare so SQLite can seek in the (timestamp, id) index
            statement = statement.where(
                tuple_(table.c.timestamp, table.c.id) < tuple_(timestamp, record_id)
            )

    statement = statement.order_by(table.c.timestamp.desc(), table.c.id.desc())
    if not paginated:
        rows = session.execute(statement).all()
        return [_history_item(row) for row in rows]

    page_limit = limit if limit is not None else 100
    rows = session.execute(statement.limit(page_limit + 1)).all()
    has_more = len(rows) > page_limit
    items = [_history_item(row) for row in rows[:page_limit]]
    next_cursor = None
    if has_more and items:
        last_item = items[-1]
        next_cursor = f"{last_item['timestamp']}|{last_item['id']}"
    return {"items": items, "next_cursor": next_cursor}


def _first_seen(session: Session, column, skill_value):
    """Where the old loop first met this skill: newest row, then position in its list."""
    table = AnalysisRecord.__table__
    item = func.json_each(column).table_valued("key", "value").alias("item")
    position = item.c.key.cast(Integer)
    row = session.execute(
        select(table.c.timestamp, table.c.id, position.label("position"))
        .select_from(table.join(item, true()))
        .where(func.json_type(column) == "array", item.c.value == skill_value)
        .order_by(table.c.timestamp.desc(), table.c.id.desc(), position)
        .limit(1)
    ).one()
    # sorted with reverse=True: newest first (NULL timestamps last like the old DESC sort),
    # then higher id, then lower list position
    return (row.timestamp is not None, row.timestamp or datetime.min, row.id, -row.position)


def _top_skills(session: Session, column_name, top_n=5):
    table = AnalysisRecord.__table__
    column = table.c[column_name]
    item = func.json_each(column).table_valued("key", "value").alias("item")
    counts = session.execute(
        select(item.c.value.label("skill"), func.count().label("count"))
        .select_from(table.join(item, true()))
        # a None list is stored as JSON null, the old code skipped those rows
        .where(func.json_type(column) == "array")
        .group_by(item.c.value)
    ).all()
    if not counts:
        return []

    # the old code sorted by count and kept first seen order for ties,
    # so only skills tied at the cutoff need their first position looked up
    ordered = sorted(counts, key=lambda row: row.count, reverse=True)
    cutoff = ordered[min(top_n, len(ordered)) - 1].count
    candidates = [row for row in ordered if row.count >= cutoff]
    tied_counts = {
        count for count in {row.count for row in candidates}
        if sum(1 for row in candidates if row.count == count) > 1
    }
    first_seen = {
        row.skill: _first_seen(session, column, row.skill)
        for row in candidates if row.count in tied_counts
    }

    top = []
    for count in sorted({row.count for row in candidates}, reverse=True):
        group = [row for row in candidates if row.count == count]
        if count in tied_counts:
            group.sort(key=lambda row: first_seen[row.skill], reverse=True)
        top.extend(group)
    return [{"skill": row.skill, "count": row.count} for row in top[:top_n]]


def get_stats(session: Session):
    table = AnalysisRecord.__table__
    aggregate = session.execute(select(
        func.count().label("total_analyses"),
        func.max(table.c.similarity_score).label("best_score"),
        func.avg(table.c.similarity_score).label("avg_score"),
    )).one()
    if not aggregate.total_analyses:
        return {
            "avg_score": 0.0,
            "best_score": 0.0,
            "recent_avg": 0.0,
            "total_analyses": 0,
            "top_missing_skills": [],
            "top_matched_skills": [],
            "last_analysis_at": None,
        }

    newest = session.execute(select(
        table.c.timestamp,
        table.c.similarity_score,
    ).order_by(table.c.timestamp.desc(), table.c.id.desc()).limit(5)).all()
    # SQLite AVG can round differently from Python sum, so avg_score comparisons allow 1e-12.
    recent_avg = sum(row.similarity_score for row in newest) / len(newest)
    return {
        "avg_score": aggregate.avg_score,
        "best_score": aggregate.best_score,
        "recent_avg": recent_avg,
        "total_analyses": aggregate.total_analyses,
        "top_missing_skills": _top_skills(session, "missing_skills"),
        "top_matched_skills": _top_skills(session, "matched_skills"),
        "last_analysis_at": newest[0].timestamp.isoformat() if newest[0].timestamp else None,
    }
