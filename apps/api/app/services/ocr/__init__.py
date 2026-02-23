from app.services.ocr.paddle import PaddleOCRService, SuryaOCRService, OCRResult
from app.services.ocr.postprocess import LegalPostProcessor, LegalEntity

__all__ = [
    "PaddleOCRService",
    "SuryaOCRService",
    "OCRResult",
    "LegalPostProcessor",
    "LegalEntity",
]
