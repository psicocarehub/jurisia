"""
Law Change Detector: monitors for new legislation, amendments,
revocations, new sumulas, and binding theses from repetitive appeals.

Detects changes by comparing newly ingested DOU documents against
existing legislation in the database and RAG index.

Includes temporal law versioning: maintains a history of article versions
allowing queries like "qual era o Art. 5 do CPC em 2020?".
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import settings
from app.services.ingestion.lexml import LexMLClient

logger = logging.getLogger("jurisai.law_change")


class ChangeType(str, Enum):
    NEW_LAW = "new_law"
    AMENDMENT = "amendment"
    REVOCATION = "revocation"
    NEW_SUMULA = "new_sumula"
    SUMULA_CANCELLED = "sumula_cancelled"
    NEW_THESIS = "new_thesis"
    ARTICLE_ADDED = "article_added"
    ARTICLE_MODIFIED = "article_modified"
    ARTICLE_REVOKED = "article_revoked"


AREA_KEYWORDS: dict[str, list[str]] = {
    "constitucional": ["constituição", "constitucional", "direito fundamental", "emenda constitucional"],
    "civil": ["código civil", "obrigações", "contratos", "responsabilidade civil", "família", "sucessões"],
    "penal": ["código penal", "crime", "pena", "execução penal", "lei penal"],
    "trabalhista": ["CLT", "trabalho", "trabalhista", "empregado", "empregador", "férias"],
    "tributario": ["tributário", "imposto", "taxa", "contribuição", "ICMS", "ISS", "IR"],
    "processual_civil": ["CPC", "processo civil", "recurso", "execução", "tutela"],
    "processual_penal": ["CPP", "processo penal", "inquérito", "prisão"],
    "administrativo": ["licitação", "administrativo", "servidor público", "concurso"],
    "consumidor": ["CDC", "consumidor", "fornecedor", "defeito", "vício"],
    "ambiental": ["ambiental", "meio ambiente", "licenciamento", "IBAMA"],
    "empresarial": ["empresa", "sociedade", "falência", "recuperação judicial"],
    "previdenciario": ["previdência", "INSS", "aposentadoria", "benefício"],
}


@dataclass
class LawChange:
    """Represents a detected change in legislation."""

    change_type: ChangeType
    title: str
    description: str
    affected_law: str
    affected_articles: list[str] = field(default_factory=list)
    new_law_reference: str = ""
    areas: list[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical
    source_url: str = ""
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class LawChangeDetector:
    """Detects legislative changes from newly ingested documents."""

    REVOCATION_PATTERNS = [
        r"(?:fica|ficam)\s+revogad[oa]s?\s+(.*?)(?:\.|;)",
        r"revoga(?:-se|m-se)?\s+(.*?)(?:\.|;)",
        r"(?:são|fica)\s+expressamente\s+revogad[oa]s?\s+(.*?)(?:\.|;)",
    ]

    AMENDMENT_PATTERNS = [
        r"(?:altera|modifica)\s+(?:a redação d[oa]\s+)?(.*?)(?:\.|;)",
        r"(?:dá|dão)\s+nova\s+redação\s+(?:ao?|à)\s+(.*?)(?:\.|;)",
        r"(?:acrescenta|inclui)\s+(?:o\s+)?(?:art|§|inciso|alínea|parágrafo)\s*(.*?)(?:\.|;)",
        r"(?:o|a)\s+(?:art|§)\.\s*\d+.*?(?:da|do)\s+(Lei|Decreto|Código)\s+.*?passa\s+a\s+vigorar",
    ]

    SUMULA_PATTERNS = [
        r"Súmula\s+(?:Vinculante\s+)?(?:n[ºo°]?\s*)?(\d+)",
        r"SÚMULA\s+(?:VINCULANTE\s+)?(?:N[ºO°]?\s*)?(\d+)",
    ]

    THESIS_PATTERNS = [
        r"(?:Tese\s+firmada|Tese\s+fixada|Tema\s+(?:n[ºo°]?\s*)?\d+)\s*:?\s*(.*?)(?:\.|$)",
        r"recurso[s]?\s+repetitivo[s]?\s*.*?(?:firmou|fixou)\s+(?:a\s+)?tese\s*:?\s*(.*?)(?:\.|$)",
    ]

    def __init__(self) -> None:
        self.lexml = LexMLClient()

    async def detect_changes(
        self,
        documents: list[dict[str, Any]],
    ) -> list[LawChange]:
        """
        Analyze a batch of newly ingested DOU documents for legislative changes.

        Args:
            documents: List of dicts with keys: doc_type, title, ementa, full_text,
                       publication_date, source_url
        """
        changes: list[LawChange] = []

        for doc in documents:
            doc_changes = await self._analyze_document(doc)
            changes.extend(doc_changes)

        self._assign_severity(changes)
        return changes

    async def _analyze_document(self, doc: dict[str, Any]) -> list[LawChange]:
        changes: list[LawChange] = []
        text = (doc.get("ementa", "") or "") + "\n" + (doc.get("full_text", "") or "")
        title = doc.get("title", "")
        doc_type = doc.get("doc_type", "")
        source_url = doc.get("source_url", "")
        areas = self._classify_areas(text)

        if doc_type in ("lei", "lei_complementar", "decreto", "medida_provisoria", "emenda_constitucional"):
            changes.append(LawChange(
                change_type=ChangeType.NEW_LAW,
                title=f"Nova legislação: {title}",
                description=doc.get("ementa", title)[:500],
                affected_law=title,
                areas=areas,
                new_law_reference=title,
                source_url=source_url,
            ))

        revocations = self._detect_revocations(text)
        for rev in revocations:
            changes.append(LawChange(
                change_type=ChangeType.REVOCATION,
                title=f"Revogação detectada: {rev['law']}",
                description=f"{title} revoga {rev['law']}. Trecho: {rev['context'][:300]}",
                affected_law=rev["law"],
                affected_articles=rev.get("articles", []),
                new_law_reference=title,
                areas=areas,
                source_url=source_url,
            ))

        amendments = self._detect_amendments(text)
        for amd in amendments:
            changes.append(LawChange(
                change_type=ChangeType.AMENDMENT,
                title=f"Alteração detectada: {amd['law']}",
                description=f"{title} altera {amd['law']}. Trecho: {amd['context'][:300]}",
                affected_law=amd["law"],
                affected_articles=amd.get("articles", []),
                new_law_reference=title,
                areas=areas,
                source_url=source_url,
            ))

        sumulas = self._detect_sumulas(text, title)
        changes.extend(sumulas)

        theses = self._detect_theses(text, title, areas, source_url)
        changes.extend(theses)

        article_changes = self._detect_article_changes(text, title, areas, source_url)
        changes.extend(article_changes)

        return changes

    def _detect_revocations(self, text: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for pattern in self.REVOCATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                context = match.group(0)
                laws = self._extract_law_references(match.group(1))
                articles = self._extract_article_numbers(context)
                for law in laws:
                    results.append({"law": law, "context": context, "articles": articles})
        return results

    def _detect_amendments(self, text: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for pattern in self.AMENDMENT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                context = match.group(0)
                affected = match.group(1) if match.lastindex else context
                laws = self._extract_law_references(affected)
                articles = self._extract_article_numbers(context)
                for law in laws:
                    results.append({"law": law, "context": context, "articles": articles})
        return results

    def _detect_sumulas(self, text: str, title: str) -> list[LawChange]:
        changes: list[LawChange] = []
        for pattern in self.SUMULA_PATTERNS:
            for match in re.finditer(pattern, text):
                number = match.group(1)
                is_vinculante = "vinculante" in match.group(0).lower()
                label = "Vinculante " if is_vinculante else ""

                ct = ChangeType.NEW_SUMULA
                if re.search(r"cancel", text[max(0, match.start() - 100):match.end() + 100], re.IGNORECASE):
                    ct = ChangeType.SUMULA_CANCELLED

                tribunal = ""
                for t in ["STF", "STJ", "TST", "TSE"]:
                    if t in text[max(0, match.start() - 50):match.end() + 50]:
                        tribunal = t
                        break

                changes.append(LawChange(
                    change_type=ct,
                    title=f"Súmula {label}{number} {tribunal}".strip(),
                    description=f"Detectada referência a Súmula {label}{number} do {tribunal} em: {title}",
                    affected_law=f"Súmula {label}{number} {tribunal}".strip(),
                    areas=self._classify_areas(text),
                    severity="high" if is_vinculante else "medium",
                ))
        return changes

    def _detect_theses(
        self, text: str, title: str, areas: list[str], source_url: str,
    ) -> list[LawChange]:
        changes: list[LawChange] = []
        for pattern in self.THESIS_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                thesis_text = match.group(1).strip() if match.lastindex else match.group(0).strip()
                changes.append(LawChange(
                    change_type=ChangeType.NEW_THESIS,
                    title=f"Tese firmada: {thesis_text[:100]}",
                    description=thesis_text[:500],
                    affected_law=title,
                    areas=areas,
                    source_url=source_url,
                    severity="high",
                ))
        return changes

    def _detect_article_changes(
        self, text: str, title: str, areas: list[str], source_url: str,
    ) -> list[LawChange]:
        changes: list[LawChange] = []

        added = re.findall(
            r"(?:acrescenta|inclui)\s+(?:o\s+)?Art\.\s*(\d+\S*)\s+(?:à|ao|da|do)\s+(.*?)(?:\.|;)",
            text, re.IGNORECASE,
        )
        for art_num, law_ref in added:
            changes.append(LawChange(
                change_type=ChangeType.ARTICLE_ADDED,
                title=f"Art. {art_num} acrescentado a {law_ref.strip()[:100]}",
                description=f"{title} acrescenta Art. {art_num} a {law_ref.strip()[:200]}",
                affected_law=law_ref.strip()[:200],
                affected_articles=[art_num],
                new_law_reference=title,
                areas=areas,
                source_url=source_url,
            ))

        modified = re.findall(
            r"Art\.\s*(\d+\S*)\s+(?:da|do)\s+(.*?)\s+passa\s+a\s+vigorar",
            text, re.IGNORECASE,
        )
        for art_num, law_ref in modified:
            changes.append(LawChange(
                change_type=ChangeType.ARTICLE_MODIFIED,
                title=f"Art. {art_num} de {law_ref.strip()[:100]} modificado",
                description=f"{title} modifica Art. {art_num} de {law_ref.strip()[:200]}",
                affected_law=law_ref.strip()[:200],
                affected_articles=[art_num],
                new_law_reference=title,
                areas=areas,
                source_url=source_url,
            ))

        revoked = re.findall(
            r"(?:revoga|revogado)\s+(?:o\s+)?Art\.\s*(\d+\S*)\s+(?:da|do)\s+(.*?)(?:\.|;)",
            text, re.IGNORECASE,
        )
        for art_num, law_ref in revoked:
            changes.append(LawChange(
                change_type=ChangeType.ARTICLE_REVOKED,
                title=f"Art. {art_num} de {law_ref.strip()[:100]} revogado",
                description=f"{title} revoga Art. {art_num} de {law_ref.strip()[:200]}",
                affected_law=law_ref.strip()[:200],
                affected_articles=[art_num],
                new_law_reference=title,
                areas=areas,
                source_url=source_url,
            ))

        return changes

    def _extract_law_references(self, text: str) -> list[str]:
        """Extract law references like 'Lei 10.406/2002' from text."""
        pattern = r"((?:Lei|Decreto|Código|Resolução|Medida Provisória|Lei Complementar)\s+(?:n[ºo°]?\s*)?[\d.]+(?:/\d{4})?)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [m.strip() for m in matches] if matches else [text.strip()[:200]]

    def _extract_article_numbers(self, text: str) -> list[str]:
        return re.findall(r"Art\.\s*(\d+[°ºª]?(?:-[A-Z])?)", text, re.IGNORECASE)

    def _classify_areas(self, text: str) -> list[str]:
        """Classify which legal areas are affected based on keyword matching."""
        text_lower = text.lower()
        areas: list[str] = []
        for area, keywords in AREA_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    areas.append(area)
                    break
        return areas or ["geral"]

    def _assign_severity(self, changes: list[LawChange]) -> None:
        """Assign severity based on change type and scope."""
        for change in changes:
            if change.severity != "medium":
                continue

            if change.change_type == ChangeType.REVOCATION:
                change.severity = "critical"
            elif change.change_type in (ChangeType.NEW_SUMULA, ChangeType.NEW_THESIS):
                change.severity = "high"
            elif change.change_type == ChangeType.AMENDMENT:
                change.severity = "high" if len(change.affected_articles) > 3 else "medium"
            elif change.change_type == ChangeType.NEW_LAW:
                if "emenda_constitucional" in change.affected_law.lower():
                    change.severity = "critical"
                else:
                    change.severity = "medium"


@dataclass
class ArticleVersion:
    """A specific version of a law article at a point in time."""

    law: str
    article: str
    text: str
    effective_from: str
    effective_until: str | None = None
    modified_by: str = ""
    change_type: str = ""
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TemporalLawVersioning:
    """
    Temporal versioning of law articles.

    Maintains a history of article versions so the system can answer
    questions like "qual era o Art. 5 do CPC em 2020?" and track the
    evolution of legislation over time.

    Stores versions in Supabase (table: law_article_versions) and
    maintains a local JSON cache for fast queries.
    """

    TABLE = "law_article_versions"

    def __init__(self, cache_dir: str = "/tmp/jurisai_law_versions") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, list[ArticleVersion]] = {}

    def _version_key(self, law: str, article: str) -> str:
        return f"{law.lower().strip()}::art.{article.strip()}"

    async def record_version(
        self,
        law: str,
        article: str,
        text: str,
        effective_from: str,
        modified_by: str = "",
        change_type: str = "",
        source_url: str = "",
    ) -> ArticleVersion:
        """
        Record a new version of an article.

        Closes the previous version (sets effective_until) and creates
        the new one.
        """
        key = self._version_key(law, article)
        version = ArticleVersion(
            law=law,
            article=article,
            text=text,
            effective_from=effective_from,
            modified_by=modified_by,
            change_type=change_type,
            source_url=source_url,
        )

        if key not in self._versions:
            self._versions[key] = []

        existing = self._versions[key]
        if existing:
            existing[-1].effective_until = effective_from

        self._versions[key].append(version)

        await self._persist_version(version)
        self._save_cache(key)

        logger.info(
            "Versao registrada: %s Art. %s (vigente desde %s, alterado por %s)",
            law, article, effective_from, modified_by or "original",
        )
        return version

    async def get_version_at_date(
        self,
        law: str,
        article: str,
        date: str,
    ) -> ArticleVersion | None:
        """
        Get the version of an article that was effective at a specific date.

        Args:
            law: Law identifier (e.g., "CPC", "Lei 10.406/2002")
            article: Article number (e.g., "5", "475-J")
            date: Date to query (YYYY-MM-DD)
        """
        key = self._version_key(law, article)
        versions = self._versions.get(key)

        if not versions:
            versions = await self._load_from_db(law, article)
            if versions:
                self._versions[key] = versions
            else:
                return None

        target = datetime.strptime(date, "%Y-%m-%d")

        for v in reversed(versions):
            v_from = datetime.strptime(v.effective_from, "%Y-%m-%d")
            v_until = (
                datetime.strptime(v.effective_until, "%Y-%m-%d")
                if v.effective_until
                else datetime.max
            )
            if v_from <= target < v_until:
                return v

        return None

    async def get_all_versions(
        self,
        law: str,
        article: str,
    ) -> list[ArticleVersion]:
        """Get the complete version history of an article."""
        key = self._version_key(law, article)
        versions = self._versions.get(key)

        if not versions:
            versions = await self._load_from_db(law, article)
            if versions:
                self._versions[key] = versions

        return versions or []

    async def get_current_version(
        self,
        law: str,
        article: str,
    ) -> ArticleVersion | None:
        """Get the currently effective version of an article."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return await self.get_version_at_date(law, article, today)

    async def record_changes_from_detector(
        self,
        changes: list[LawChange],
    ) -> int:
        """
        Process LawChange events from LawChangeDetector and create
        article versions automatically.
        """
        recorded = 0
        today = datetime.utcnow().strftime("%Y-%m-%d")

        for change in changes:
            if change.change_type not in (
                ChangeType.ARTICLE_ADDED,
                ChangeType.ARTICLE_MODIFIED,
                ChangeType.ARTICLE_REVOKED,
                ChangeType.AMENDMENT,
            ):
                continue

            for art in change.affected_articles:
                change_text = change.description
                if change.change_type == ChangeType.ARTICLE_REVOKED:
                    change_text = f"[REVOGADO] {change.description}"

                await self.record_version(
                    law=change.affected_law,
                    article=art,
                    text=change_text,
                    effective_from=today,
                    modified_by=change.new_law_reference,
                    change_type=change.change_type.value,
                    source_url=change.source_url,
                )
                recorded += 1

        if recorded:
            logger.info("Temporal versioning: %d versoes registradas", recorded)
        return recorded

    async def diff_versions(
        self,
        law: str,
        article: str,
        date1: str,
        date2: str,
    ) -> dict[str, Any] | None:
        """Compare article text between two dates."""
        v1 = await self.get_version_at_date(law, article, date1)
        v2 = await self.get_version_at_date(law, article, date2)

        if not v1 and not v2:
            return None

        return {
            "law": law,
            "article": article,
            "date1": date1,
            "date2": date2,
            "version1": {
                "text": v1.text if v1 else "[nao existia]",
                "effective_from": v1.effective_from if v1 else None,
                "modified_by": v1.modified_by if v1 else None,
            },
            "version2": {
                "text": v2.text if v2 else "[nao existia]",
                "effective_from": v2.effective_from if v2 else None,
                "modified_by": v2.modified_by if v2 else None,
            },
            "changed": (v1.text if v1 else "") != (v2.text if v2 else ""),
        }

    async def _persist_version(self, version: ArticleVersion) -> None:
        """Save version to Supabase."""
        try:
            from app.db.supabase_client import supabase_db
            await supabase_db.insert(self.TABLE, {
                "law": version.law,
                "article": version.article,
                "text": version.text[:10000],
                "effective_from": version.effective_from,
                "effective_until": version.effective_until,
                "modified_by": version.modified_by,
                "change_type": version.change_type,
                "source_url": version.source_url,
                "metadata": version.metadata,
            })
        except Exception as e:
            logger.warning("Erro persistindo versao no Supabase: %s", e)

    async def _load_from_db(
        self,
        law: str,
        article: str,
    ) -> list[ArticleVersion]:
        """Load versions from Supabase."""
        try:
            from app.db.supabase_client import supabase_db
            rows = await supabase_db.select(
                self.TABLE,
                filters={"law": law, "article": article},
            )
            versions = [
                ArticleVersion(
                    law=r["law"],
                    article=r["article"],
                    text=r["text"],
                    effective_from=r["effective_from"],
                    effective_until=r.get("effective_until"),
                    modified_by=r.get("modified_by", ""),
                    change_type=r.get("change_type", ""),
                    source_url=r.get("source_url", ""),
                    metadata=r.get("metadata", {}),
                )
                for r in rows
            ]
            versions.sort(key=lambda v: v.effective_from)
            return versions
        except Exception as e:
            logger.warning("Erro carregando versoes do Supabase: %s", e)
            return []

    def _save_cache(self, key: str) -> None:
        """Save versions to local JSON cache."""
        try:
            cache_file = self._cache_dir / f"{key.replace('::', '__')}.json"
            versions = self._versions.get(key, [])
            data = [
                {
                    "law": v.law,
                    "article": v.article,
                    "text": v.text[:5000],
                    "effective_from": v.effective_from,
                    "effective_until": v.effective_until,
                    "modified_by": v.modified_by,
                    "change_type": v.change_type,
                }
                for v in versions
            ]
            cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.debug("Cache save failed for %s: %s", key, e)
