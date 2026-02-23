"""
Azure Document Intelligence — fallback OCR.

Serviço de OCR usando Azure Document Intelligence quando
PaddleOCR ou Surya não estão disponíveis ou para documentos
que se beneficiam do reconhecimento da Microsoft.
"""

from typing import Optional

from app.config import settings
from app.services.ocr.paddle import OCRResult


class AzureDocIntelligenceService:
    """
    Serviço de OCR via Azure Document Intelligence.

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
        Processa PDF usando Azure Document Intelligence.

        Args:
            file_path: Caminho para o arquivo PDF
            model_id: Modelo do Document Intelligence
                      (prebuilt-read, prebuilt-layout, etc.)

        Returns:
            OCRResult com texto extraído, confiança e metadados
        """
        if not self.api_key or not self.endpoint:
            return OCRResult(
                text="",
                confidence=0.0,
                pages=0,
                structured_output=None,
            )

        # Stub: implementação real via azure-ai-documentintelligence
        # from azure.ai.documentintelligence import DocumentIntelligenceClient
        # from azure.core.credentials import AzureKeyCredential
        # client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
        # with open(file_path, "rb") as f:
        #     poller = client.begin_analyze_document(model_id, document=f)
        # result = poller.result()
        _ = file_path, model_id

        return OCRResult(
            text="",
            confidence=0.0,
            pages=0,
            structured_output=None,
        )
