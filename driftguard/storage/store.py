from __future__ import annotations
import json
from datetime import datetime
from typing import Optional
from sqlmodel import Field, Session, SQLModel, create_engine,select
from driftguard.ingest.log_schema import DriftLogEntry, MetricPoint


class LogRow(SQLModel, table=True):
    __tablename__ = "log_rows"
 
    log_id: str = Field(primary_key=True)
    timestamp: datetime = Field(index=True)
    integration_path: str
    provider: str = Field(index=True)
    model_id: str = Field(index=True)
 
    prompt: str
    response_text: Optional[str] = None
    request_params_json: str = "{}"  # dict[str, Any] serialized
 
    outcome: str
    error_message: Optional[str] = None
 
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
 
    eval_run_id: Optional[str] = Field(default=None, index=True)
    eval_task_type: Optional[str] = None
    eval_task_id: Optional[str] = None
    eval_score: Optional[float] = None

class MetricSeriesRow(SQLModel, table=True):
    __tablename__ = "metric_series"
 
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    timestamp: datetime = Field(index=True)
    provider: str = Field(index=True)
    model_id: str = Field(index=True)
 
    accuracy: Optional[float] = None
    format_adherence: Optional[float] = None
    refusal_rate: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
 
    n_calls: int = 0

class ChangePointEventRow(SQLModel, table=True):
    __tablename__ = "changepoint_events"
 
    id: Optional[int] = Field(default=None, primary_key=True)
    detected_at: datetime  # wall-clock time the detector produced this event
    provider: str = Field(index=True)
    model_id: str = Field(index=True)
    metric_name: str  # "accuracy" | "format_adherence" | "refusal_rate" | "latency_p95_ms"
 
    detector: str  # "pelt" | "sequential" | "baseline"
    changepoint_timestamp: Optional[datetime] = None  # estimated time of the shift itself
    pre_mean: Optional[float] = None
    post_mean: Optional[float] = None
 
    p_value_raw: Optional[float] = None
    p_value_corrected: Optional[float] = None  # after Benjamini-Hochberg
    significant: bool = False
 
    drift_type: Optional[str] = None       # set by classification stage
    remediation: Optional[str] = None      # set by remediation stage
 
    reviewed: bool = False  # human-in-the-loop review flag


class DriftStore:
    """Thin persistence facade. One instance per process is fine (SQLite)."""
 
    def __init__(self, db_path: str = "driftguard.db", echo: bool = False):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=echo)
        SQLModel.metadata.create_all(self.engine)
 
    # -- writes ----------------------------------------------------------
 
    def write_log(self, entry: DriftLogEntry) -> None:
        row = LogRow(
            log_id=entry.log_id,
            timestamp=entry.timestamp,
            integration_path=entry.integration_path,
            provider=entry.provider,
            model_id=entry.model_id,
            prompt=entry.prompt,
            response_text=entry.response_text,
            request_params_json=json.dumps(entry.request_params),
            outcome=entry.outcome,
            error_message=entry.error_message,
            latency_ms=entry.latency_ms,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            eval_run_id=entry.eval_run_id,
            eval_task_type=entry.eval_task_type,
            eval_task_id=entry.eval_task_id,
            eval_score=entry.eval_score,
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
 
    def write_metric_point(self, point: MetricPoint) -> None:
        row = MetricSeriesRow(**point.model_dump())
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
 
    def write_changepoint_event(self, event: ChangePointEventRow) -> ChangePointEventRow:
        with Session(self.engine) as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def get_metric_series(
            self, provider:str, model_id:str, metric_name: Optional[str]=None
    )->list[MetricSeriesRow]:
        with Session(self.engine) as session:
            stmt = (
                select(MetricSeriesRow)
                .where(MetricSeriesRow.provider == provider)
                .where(MetricSeriesRow.model_id == model_id)
                .order_by(MetricSeriesRow.timestamp)
            )
            return list(session.exec(stmt).all())
 
    def get_logs_for_run(self, eval_run_id: str) -> list[LogRow]:
        with Session(self.engine) as session:
            stmt = select(LogRow).where(LogRow.eval_run_id == eval_run_id)
            return list(session.exec(stmt).all())
 
    def get_exemplars(
        self, provider: str, model_id: str, start: datetime, end: datetime, limit: int = 5
    ) -> list[LogRow]:
        """Pull sample prompt/response pairs from a flagged window, for the dashboard."""
        with Session(self.engine) as session:
            stmt = (
                select(LogRow)
                .where(LogRow.provider == provider)
                .where(LogRow.model_id == model_id)
                .where(LogRow.timestamp >= start)
                .where(LogRow.timestamp <= end)
                .limit(limit)
            )
            return list(session.exec(stmt).all())
 
    def get_open_events(self) -> list[ChangePointEventRow]:
        with Session(self.engine) as session:
            stmt = (
                select(ChangePointEventRow)
                .where(ChangePointEventRow.significant == True)  # noqa: E712
                .where(ChangePointEventRow.reviewed == False)  # noqa: E712
                .order_by(ChangePointEventRow.detected_at.desc())
            )
            return list(session.exec(stmt).all())
 