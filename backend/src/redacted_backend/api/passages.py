from __future__ import annotations

import hashlib
import json
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from redacted_backend.db.base import Database
from redacted_backend.db.repositories.passages import (
    PassageConflictError,
    PassageNotFoundError,
    PassagePageOutOfRangeError,
    PassageRepository,
)
from redacted_backend.domain.passages import (
    PassageMergeResult,
    PassageRecord,
    PassageRect,
    PassageSource,
    PassageSplitResult,
)


class CreatePassageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=200)
    page_number: int = Field(ge=1)
    rects: list[PassageRect] = Field(min_length=1, max_length=50)


class PatchPassageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, min_length=1, max_length=200)
    page_number: int | None = Field(default=None, ge=1)
    rects: list[PassageRect] | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )

    @model_validator(mode="after")
    def contains_a_change(self) -> PatchPassageRequest:
        if self.label is None and self.page_number is None and self.rects is None:
            raise ValueError("at least one passage field is required")
        return self


class MergePassagesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_ids: list[UUID] = Field(min_length=2, max_length=50)

    @model_validator(mode="after")
    def contains_unique_passages(self) -> MergePassagesRequest:
        if len(set(self.passage_ids)) != len(self.passage_ids):
            raise ValueError("passage_ids must be unique")
        return self


class SplitPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=200)
    page_number: int = Field(ge=1)
    rects: list[PassageRect] = Field(min_length=1, max_length=50)


class SplitPassageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[SplitPart] = Field(min_length=2, max_length=50)


def create_passages_router(database: Database | None) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/documents/{document_id}/passages",
        response_model=PassageRecord,
        status_code=status.HTTP_201_CREATED,
    )
    def create_passage(
        document_id: UUID,
        request: CreatePassageRequest,
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            min_length=1,
            max_length=255,
        ),
    ) -> PassageRecord:
        ready_database = _require_database(database)
        scope = f"create-passage:{document_id}"
        request_hash = _request_hash(request)
        try:
            with ready_database.session_factory() as session, session.begin():
                repository = PassageRepository(session)
                if idempotency_key is not None:
                    existing = repository.get_idempotent(
                        scope=scope,
                        key=idempotency_key,
                        request_hash=request_hash,
                    )
                    if existing is not None:
                        return existing
                created = repository.create(
                    document_id=document_id,
                    label=request.label,
                    page_number=request.page_number,
                    rects=request.rects,
                    source=PassageSource.MANUAL,
                )
                if idempotency_key is not None:
                    repository.save_idempotency(
                        scope=scope,
                        key=idempotency_key,
                        request_hash=request_hash,
                        resource_id=created.id,
                    )
                return created
        except PassageNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except PassageConflictError as error:
            raise _conflict(error) from error
        except PassagePageOutOfRangeError as error:
            raise _page_out_of_range(error) from error

    @router.patch("/passages/{passage_id}", response_model=PassageRecord)
    def patch_passage(
        passage_id: UUID,
        request: PatchPassageRequest,
    ) -> PassageRecord:
        ready_database = _require_database(database)
        try:
            with ready_database.session_factory() as session, session.begin():
                return PassageRepository(session).patch(
                    passage_id,
                    label=request.label,
                    page_number=request.page_number,
                    rects=request.rects,
                )
        except PassageNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except PassagePageOutOfRangeError as error:
            raise _page_out_of_range(error) from error

    @router.delete("/passages/{passage_id}", response_model=PassageRecord)
    def delete_passage(passage_id: UUID) -> PassageRecord:
        ready_database = _require_database(database)
        try:
            with ready_database.session_factory() as session, session.begin():
                return PassageRepository(session).mark_deleted(passage_id)
        except PassageNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error

    @router.post("/passages:merge", response_model=PassageMergeResult)
    def merge_passages(request: MergePassagesRequest) -> PassageMergeResult:
        ready_database = _require_database(database)
        try:
            with ready_database.session_factory() as session, session.begin():
                passage, merged_ids = PassageRepository(session).merge(
                    request.passage_ids
                )
                return PassageMergeResult(
                    passage=passage,
                    merged_ids=merged_ids,
                )
        except PassageNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except PassageConflictError as error:
            raise _conflict(error) from error

    @router.post(
        "/passages/{passage_id}:split",
        response_model=PassageSplitResult,
    )
    def split_passage(
        passage_id: UUID,
        request: SplitPassageRequest,
    ) -> PassageSplitResult:
        ready_database = _require_database(database)
        try:
            with ready_database.session_factory() as session, session.begin():
                passages = PassageRepository(session).split(
                    passage_id,
                    parts=[
                        (part.label, part.page_number, part.rects)
                        for part in request.parts
                    ],
                )
                return PassageSplitResult(passages=passages)
        except PassageNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except PassagePageOutOfRangeError as error:
            raise _page_out_of_range(error) from error

    return router


def _require_database(database: Database | None) -> Database:
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "database_not_configured"},
        )
    return database


def _request_hash(request: BaseModel) -> str:
    canonical = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _conflict(error: PassageConflictError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "passage_conflict", "message": str(error)},
    )


def _page_out_of_range(error: PassagePageOutOfRangeError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"code": "passage_page_out_of_range", "message": str(error)},
    )
