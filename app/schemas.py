from typing import Any

from pydantic import BaseModel, ConfigDict


class ModelJobPayload(BaseModel):
    kind: str
    selection: list[str] | None = None


class PersonPayload(BaseModel):
    name: str


class PersonFilePayload(BaseModel):
    person: str
    file: str


class PipelineTogglePayload(BaseModel):
    enabled: bool


class UnknownAssignPayload(BaseModel):
    person: str
    files: list[str]


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Prediction(BoundingBox):
    label: str
    confidence: float
    userid: str | None = None


class VisionResponse(BaseModel):
    success: bool
    message: str
    count: int | None = 0
    predictions: list[Prediction]
    inferenceMs: int
    processMs: int
    analysisRoundTripMs: int
    moduleName: str = "OBF Vision"
    moduleId: str = "OBF"
    code: int = 200
    command: str
    requestId: str | None = None
    inferenceDevice: str | None = None
    timestampUTC: str | None = None


class WsStatusData(BaseModel):
    pipeline_enabled: bool
    uptime: str
    req_count: int
    ema_ms: float
    last_ms: float = 0.0
    fps: float
    last_labels: list[str]

