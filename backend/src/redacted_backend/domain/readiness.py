from pydantic import BaseModel, ConfigDict


class ComponentReadiness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ready: bool
    detail: str


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    http: ComponentReadiness
    database: ComponentReadiness
    worker: ComponentReadiness
    corpus: ComponentReadiness
    model: ComponentReadiness
    prediction_ready: bool
