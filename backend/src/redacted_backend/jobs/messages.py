from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AnalysisJobMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: UUID
