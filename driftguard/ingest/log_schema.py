from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any,Optional
from pydantic import BaseModel, Field, field_validator

class IntegrationPath(str,Enum):
    WRAPPER="wrapper"
    PROXY="proxy"
    EVAL_SUITE='eval_suite'

class CallOutcome(str,Enum):
    OK="ok"
    ERROR="error"
    TIMEOUT="timeout"
    RATE_LIMITED="rate_limited"

class DriftLogEntry(BaseModel):
    #one row per llm api call, even if it is ingested from the proxy or the wrapper
    log_id:str=Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:datetime=Field(default_factory=lambda: datetime.now(timezone.utc))
    integration_path:IntegrationPath

    provider:str   #can be any of the famous llms
    model_id:str   #model requested by the user


    prompt:str
    response_text:Optional[str]=None
    request_params:dict[str,Any]=Field(default_factory=dict) #temperature, max_tokens, etc

    #outcome
    outcome:CallOutcome=CallOutcome.OK
    error_message:Optional[str]=None

    #timing
    latency_ms:Optional[float]=None

    #token accounting
    prompt_tokens: Optional[int]=None
    completion_tokens: Optional[int]=None


    #eval suite specific tagging
    eval_run_id: Optional[str]=None   #group all the logs for a single eval run together
    eval_task_type=Optional[str]=None  #can be either accuracy, format, latency or confidence
    eval_task_id=Optional[str]=None    #id of the eval_suite specific to the task
    eval_score=Optional[float]=None   #score received after going through the eval suite (one for each call)

    @field_validator("provider")
    @classmethod
    def _lower_provider(cls, v: str)->str:
        return v.strip().lower()

    model_config={
        "use_enum_values":True,
    }
class MetricPoint(BaseModel):
    run_id:str
    timestamp:datetime
    provider:str
    model_id:str

    accuracy: Optional[float]=None
    format_adherence: Optional[float]=None
    refuslal_rate: Optional[float]=None
    latency_p50_ms: Optional[float]=None
    latency_p95_ms: Optional[float]=None

    n_call:int =0
    



