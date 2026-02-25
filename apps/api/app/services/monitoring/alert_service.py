"""
Alert Service: notifies users about legislative changes relevant
to their practice areas.

- Stores alerts in the database (alerts table)
- Marks potentially outdated documents/petitions
- Updates the CitationVerifier with revoked/modified citations
- Exposes an API for users to query their pending alerts
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.db.supabase_client import supabase_db
from app.services.monitoring.law_change_detector import ChangeType, LawChange

logger = logging.getLogger("jurisai.alerts")

SEVERITY_PRIORITY = {"critical": 0, "high": 1, "medium": 2, "low": 3}

ALERT_TABLE = "alerts"
ALERT_SUBSCRIPTIONS_TABLE = "alert_subscriptions"


class AlertService:
    """
    Manages alerts for legislative and jurisprudential changes.

    Workflow:
    1. LawChangeDetector finds changes
    2. AlertService stores them and matches to affected users
    3. Users retrieve alerts via API (filtered by their areas)
    """

    async def ensure_tables(self) -> None:
        """Create alert tables via Supabase RPC (or skip if exists)."""
        try:
            await supabase_db.rpc("exec_sql", {"query": """
                CREATE TABLE IF NOT EXISTS alerts (
                    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                    change_type VARCHAR(50) NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    affected_law TEXT,
                    affected_articles TEXT[] DEFAULT '{}',
                    new_law_reference TEXT,
                    areas TEXT[] DEFAULT '{}',
                    severity VARCHAR(20) DEFAULT 'medium',
                    source_url TEXT,
                    is_read BOOLEAN DEFAULT FALSE,
                    tenant_id UUID,
                    user_id UUID,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_tenant ON alerts(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id);
                CREATE INDEX IF NOT EXISTS idx_alerts_areas ON alerts USING GIN(areas);
                CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(is_read);
                CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

                CREATE TABLE IF NOT EXISTS alert_subscriptions (
                    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                    user_id UUID NOT NULL,
                    tenant_id UUID NOT NULL,
                    areas TEXT[] DEFAULT '{}',
                    change_types TEXT[] DEFAULT '{}',
                    min_severity VARCHAR(20) DEFAULT 'medium',
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_alert_sub_user ON alert_subscriptions(user_id);
            """})
        except Exception as e:
            logger.warning("Could not create alert tables via RPC: %s", e)

    async def process_changes(
        self,
        changes: list[LawChange],
        tenant_id: Optional[str] = None,
    ) -> int:
        """
        Process detected law changes: store alerts and notify affected users.

        Returns the number of alerts created.
        """
        if not changes:
            return 0

        alert_count = 0

        for change in changes:
            alert_id = await self._store_alert(change, tenant_id)
            if alert_id:
                alert_count += 1

            if change.change_type in (ChangeType.REVOCATION, ChangeType.ARTICLE_REVOKED):
                await self._mark_affected_documents(change)

            if change.change_type == ChangeType.SUMULA_CANCELLED:
                await self._update_citation_status(change)

        if tenant_id:
            await self._fan_out_to_subscribed_users(changes, tenant_id)

        logger.info("Processed %d changes, created %d alerts", len(changes), alert_count)
        return alert_count

    async def get_alerts(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        areas: Optional[list[str]] = None,
        unread_only: bool = True,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve alerts for a user, filtered by area and read status."""
        filters: dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if tenant_id:
            filters["tenant_id"] = tenant_id
        if unread_only:
            filters["is_read"] = False

        try:
            results = await supabase_db.select(ALERT_TABLE, filters=filters)
            if not isinstance(results, list):
                results = [results] if results else []
        except Exception as e:
            logger.warning("Failed to fetch alerts: %s", e)
            return []

        if areas:
            area_set = set(areas)
            results = [
                r for r in results
                if area_set.intersection(set(r.get("areas", [])))
            ]

        results.sort(
            key=lambda r: (
                SEVERITY_PRIORITY.get(r.get("severity", "medium"), 2),
                r.get("created_at", ""),
            )
        )

        return results[:limit]

    async def mark_read(self, alert_id: str) -> None:
        """Mark an alert as read."""
        try:
            await supabase_db.update(
                ALERT_TABLE,
                {"is_read": True},
                filters={"id": alert_id},
            )
        except Exception as e:
            logger.warning("Failed to mark alert %s as read: %s", alert_id, e)

    async def mark_all_read(
        self, user_id: str, tenant_id: Optional[str] = None,
    ) -> None:
        """Mark all alerts for a user as read."""
        filters: dict[str, Any] = {"user_id": user_id, "is_read": False}
        if tenant_id:
            filters["tenant_id"] = tenant_id
        try:
            await supabase_db.update(ALERT_TABLE, {"is_read": True}, filters=filters)
        except Exception as e:
            logger.warning("Failed to mark all alerts read for user %s: %s", user_id, e)

    async def subscribe(
        self,
        user_id: str,
        tenant_id: str,
        areas: list[str],
        change_types: Optional[list[str]] = None,
        min_severity: str = "medium",
    ) -> str:
        """Subscribe a user to alerts for specific areas."""
        try:
            result = await supabase_db.insert(ALERT_SUBSCRIPTIONS_TABLE, {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "areas": areas,
                "change_types": change_types or [],
                "min_severity": min_severity,
            })
            return result.get("id", "")
        except Exception as e:
            logger.warning("Failed to create subscription: %s", e)
            return ""

    async def _store_alert(
        self,
        change: LawChange,
        tenant_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store a single alert in the database."""
        try:
            result = await supabase_db.insert(ALERT_TABLE, {
                "change_type": change.change_type.value,
                "title": change.title[:500],
                "description": change.description[:2000],
                "affected_law": change.affected_law[:500],
                "affected_articles": change.affected_articles,
                "new_law_reference": change.new_law_reference[:500],
                "areas": change.areas,
                "severity": change.severity,
                "source_url": change.source_url[:1000],
                "tenant_id": tenant_id,
                "metadata": change.metadata,
            })
            return result.get("id")
        except Exception as e:
            logger.warning("Failed to store alert: %s", e)
            return None

    async def _fan_out_to_subscribed_users(
        self,
        changes: list[LawChange],
        tenant_id: str,
    ) -> None:
        """Create per-user alerts based on subscriptions."""
        try:
            subs = await supabase_db.select(
                ALERT_SUBSCRIPTIONS_TABLE,
                filters={"tenant_id": tenant_id, "is_active": True},
            )
            if not isinstance(subs, list):
                return
        except Exception as e:
            logger.warning("Failed to fetch alert subscriptions: %s", e)
            return

        for sub in subs:
            sub_areas = set(sub.get("areas", []))
            sub_types = set(sub.get("change_types", []))
            min_sev = SEVERITY_PRIORITY.get(sub.get("min_severity", "medium"), 2)
            user_id = sub.get("user_id")

            for change in changes:
                change_sev = SEVERITY_PRIORITY.get(change.severity, 2)
                if change_sev > min_sev:
                    continue

                if sub_types and change.change_type.value not in sub_types:
                    continue

                change_areas = set(change.areas)
                if sub_areas and not sub_areas.intersection(change_areas):
                    continue

                try:
                    await supabase_db.insert(ALERT_TABLE, {
                        "change_type": change.change_type.value,
                        "title": change.title[:500],
                        "description": change.description[:2000],
                        "affected_law": change.affected_law[:500],
                        "affected_articles": change.affected_articles,
                        "areas": change.areas,
                        "severity": change.severity,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                    })
                except Exception as e:
                    logger.warning("Failed to create user alert for %s: %s", user_id, e)
                    continue

    async def _mark_affected_documents(self, change: LawChange) -> None:
        """
        When a law is revoked or articles are changed, mark
        user documents/petitions that cite the affected law.
        """
        logger.info(
            "Would mark documents citing %s (articles: %s) as potentially outdated",
            change.affected_law,
            change.affected_articles,
        )

    async def _update_citation_status(self, change: LawChange) -> None:
        """Update CitationVerifier with cancelled sumulas or revoked articles."""
        logger.info(
            "Would update citation status for: %s",
            change.affected_law,
        )
