from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import Integer, func, or_, select, true
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
            statement = statement.where(or_(
                table.c.timestamp < timestamp,
                (table.c.timestamp == timestamp) & (table.c.id < record_id),
            ))

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


def _top_skills(session: Session, column_name):
    table = AnalysisRecord.__table__
    ranked = select(
        table.c.id,
        table.c.timestamp,
        table.c[column_name].label("skills"),
        func.row_number().over(
            order_by=(table.c.timestamp.desc(), table.c.id.desc())
        ).label("row_number"),
    ).subquery("ranked_analyses")
    skill = func.json_each(ranked.c.skills).table_valued("key", "value").alias("skill")
    first_seen = func.min(func.printf(
        "%020d:%020d", ranked.c.row_number, skill.c.key.cast(Integer)
    ))
    statement = (
        select(skill.c.value.label("skill"), func.count().label("count"), first_seen.label("first_seen"))
        .select_from(ranked.join(skill, true(), isouter=False))
        # a None list is stored as JSON null, the old code skipped those rows
        .where(func.json_type(ranked.c.skills) == "array")
        .group_by(skill.c.value)
        .order_by(func.count().desc(), first_seen)
        .limit(5)
    )
    return [
        {"skill": row.skill, "count": row.count}
        for row in session.execute(statement)
    ]


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
