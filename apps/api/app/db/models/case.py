import uuid
from decimal import Decimal
from typing import Optional

from sqlalchemy import ForeignKey, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TenantMixin, TimestampMixin


class Case(Base, TenantMixin, TimestampMixin):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    cnj_number: Mapped[Optional[str]] = mapped_column(String(25), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    area: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="active")
    client_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    client_document: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    opposing_party: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    court: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    judge_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    judge_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    estimated_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(15, 2), nullable=True
    )
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default={})
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    documents = relationship("Document", back_populates="case", lazy="selectin")
    conversations = relationship("Conversation", back_populates="case", lazy="selectin")
    petitions = relationship("Petition", back_populates="case", lazy="selectin")
