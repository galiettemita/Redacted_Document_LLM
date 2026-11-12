from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


@dataclass(frozen=True)
class Database:
    engine: Engine
    session_factory: sessionmaker[Session]

    @classmethod
    def connect(cls, database_url: str) -> Database:
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        engine = create_engine(
            database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        return cls(
            engine=engine,
            session_factory=sessionmaker(
                bind=engine,
                class_=Session,
                expire_on_commit=False,
            ),
        )
