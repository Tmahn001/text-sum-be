"""SQLite persistence for submission history.

SQLite is chosen for the demo because it needs zero setup — the whole database is
a single file (submissions.db) created on first run. The DATABASE_URL env var
can point at Postgres instead (e.g. postgresql+psycopg://user:pass@host/db)
with no code changes, since we go through SQLAlchemy.
"""
import json
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import get_settings

settings = get_settings()

# check_same_thread is a SQLite-only flag needed because FastAPI serves requests
# from a threadpool. It is ignored for other databases.
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    input_text: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(128))
    # Entities and context are stored as JSON strings to keep the schema simple;
    # they are only ever read back for display, never queried on.
    entities_json: Mapped[str] = mapped_column(Text, default="[]")
    context_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    @property
    def entities(self) -> list:
        return json.loads(self.entities_json)

    @property
    def context(self) -> list:
        return json.loads(self.context_json)

    @property
    def input_preview(self) -> str:
        text = self.input_text.strip().replace("\n", " ")
        return text[:200] + ("…" if len(text) > 200 else "")


def init_db() -> None:
    """Create tables on startup if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
