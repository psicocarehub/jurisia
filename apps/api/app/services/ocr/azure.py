"""
Azure Document Intelligence — fallback OCR.

Serviço de OCR usando Azure Document Intelligence REST API quando
PaddleOCR ou Surya não estão disponíveis ou para documentos
que se beneficiam do reconhecimento da Microsoft.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from app.config import settings
from app.services.ocr.paddle import OCRResult

logger = logging.getLogger("jurisai.ocr.azure")


class AzureDocIntelligenceService:
    """
    Serviço de OCR via Azure Document Intelligence REST API.

    Usado como fallback quando os serviços primários (PaddleOCR, Surya)
    não estão disponíveis. Requer AZURE_DOC_INTELLIGENCE_KEY e
    AZURE_DOC_INTELLIGENCE_ENDPOINT configurados.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.endpoint = (
            endpoint or settings.AZURE_DOC_INTELLIGENCE_ENDPOINT
        ).rstrip("/")
        self.api_key = api_key or settings.AZURE_DOC_INTELLIGENCE_KEY

    async def process_pdf(
        self,
        file_path: str,
        model_id: str = "prebuilt-read",
    ) -> OCRResult:
        """
        Processa PDF usando Azure Document Intelligence REST API.
        Submits file, polls for result, extracts text.
        """
        if not self.api_key or not self.endpoint:
            logger.warning("Azure Doc Intelligence not configured (missing key/endpoint)")
            return OCRResult(text="", confidence=0.0, pages=0, structured_output=None)

        analyze_url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}:analyze?api-version=2024-11-30"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/pdf",
        }

        try:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(analyze_url, headers=headers, content=pdf_bytes)
                resp.raise_for_status()

                operation_url = resp.headers.get("Operation-Location", "")
                if not operation_url:
                    logger.error("Azure: No Operation-Location header returned")
                    return OCRResult(text="", confidence=0.0, pages=0, structured_output=None)

                poll_headers = {"Ocp-Apim-Subscription-Key": self.api_key}
                for _ in range(60):
                    await asyncio.sleep(2)
                    poll_resp = await client.get(operation_url, headers=poll_headers)
                    poll_resp.raise_for_status()
                    result = poll_resp.json()
                    status = result.get("status", "")
                    if status == "succeeded":
                        return self._parse_result(result)
                    elif status == "failed":
                        logger.error("Azure analyze failed: %s", result.get("error"))
                        return OCRResult(text="", confidence=0.0, pages=0, structured_output=None)

                logger.error("Azure analyze timed out after 120s polling")
                return OCRResult(text="", confidence=0.0, pages=0, structured_output=None)

        except Exception as e:
            logger.error("Azure Document Intelligence error: %s", e)
            return OCRResult(text="", confidence=0.0, pages=0, structured_output=None)

    def _parse_result(self, result: dict) -> OCRResult:
        analyze_result = result.get("analyzeResult", {})
        content = analyze_result.get("content", "")
        pages = len(analyze_result.get("pages", []))

        confidences = []
        for page in analyze_result.get("pages", []):
            for word in page.get("words", []):
                if "confidence" in word:
                    confidences.append(word["confidence"])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=content,
            confidence=round(avg_confidence, 3),
            pages=pages,
            structured_output=analyze_result,
        )
