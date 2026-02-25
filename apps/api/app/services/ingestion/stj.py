"""
STJ Dados Abertos - dadosabertos.web.stj.jus.br
Full decision text (unlike DataJud metadata-only).
"""

import csv
import io
import logging
from datetime import date
from typing import Any, AsyncGenerator, Optional

import httpx

logger = logging.getLogger("jurisai.stj")


class STJOpenDataClient:
    """Client for STJ Open Data Portal."""

    BASE_URL = "https://dadosabertos.web.stj.jus.br"

    async def stream_decisions(
        self,
        dataset: str = "acordaos",  # acordaos, decisoes_monocraticas
        date_from: Optional[date] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream decisions from STJ Open Data Portal."""
        csv_url = f"{self.BASE_URL}/dataset/{dataset}/resource/latest.csv"

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.get(csv_url)
                response.raise_for_status()
                reader = csv.DictReader(io.StringIO(response.text))
                for row in reader:
                    if date_from and row.get("data"):
                        try:
                            row_date = date.fromisoformat(row["data"][:10])
                            if row_date < date_from:
                                continue
                        except (ValueError, KeyError):
                            logger.debug("Skipping STJ row with unparseable date: %s", row.get("data"))
                    yield dict(row)
            except Exception as e:
                logger.error("STJ stream_decisions failed: %s", e)
