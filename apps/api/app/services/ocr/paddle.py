"""
OCR services: PaddleOCR-VL 1.5 and Surya + Marker.
"""

from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

from app.config import settings


class OCRResult(BaseModel):
    """OCR processing result."""

    text: str
    confidence: float
    pages: int
    structured_output: Optional[dict] = None  # Markdown or JSON


class PaddleOCRService:
    """PaddleOCR-VL 1.5: high precision, seal/stamp recognition for Brazilian docs."""

    def __init__(self, endpoint: Optional[str] = None) -> None:
        self.endpoint = endpoint or settings.PADDLE_OCR_ENDPOINT

    async def process_pdf(
        self,
        file_path: str,
        output_format: str = "markdown",  # markdown, json
        enable_seal_recognition: bool = True,
    ) -> OCRResult:
        """Process a PDF through PaddleOCR-VL 1.5."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(file_path, "rb") as f:
                response = await client.post(
                    f"{self.endpoint}/predict",
                    files={"file": (Path(file_path).name, f, "application/pdf")},
                    data={
                        "output_format": output_format,
                        "enable_seal": str(enable_seal_recognition).lower(),
                        "enable_table": "true",
                        "lang": "pt",
                    },
                )
                response.raise_for_status()
                result = response.json()

        return OCRResult(
            text=result.get("text", ""),
            confidence=float(result.get("confidence", 0.0)),
            pages=int(result.get("pages", 0)),
            structured_output=result.get("structured"),
        )


class SuryaOCRService:
    """Surya + Marker: reading order detection, multi-column documents."""

    def __init__(self, endpoint: Optional[str] = None) -> None:
        self.endpoint = endpoint or settings.SURIYA_OCR_ENDPOINT

    async def process_pdf(self, file_path: str) -> OCRResult:
        """Process a PDF through Surya OCR endpoint."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(file_path, "rb") as f:
                response = await client.post(
                    f"{self.endpoint}/convert",
                    files={"file": (Path(file_path).name, f, "application/pdf")},
                    data={"output_format": "markdown"},
                )
                response.raise_for_status()
                result = response.json()

        return OCRResult(
            text=result.get("markdown", result.get("text", "")),
            confidence=float(result.get("confidence", 0.0)),
            pages=int(result.get("pages", 0)),
            structured_output=result.get("structured"),
        )
