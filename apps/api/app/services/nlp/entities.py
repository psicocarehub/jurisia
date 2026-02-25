"""
Entity extractor — entidades estruturadas em textos jurídicos.

Combina NER (transformers) e RegEx para extrair:
- Partes do processo
- Valores monetários
- Datas
- Números de processo (CNJ)
- Referências legislativas
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional

from app.services.nlp.ner import LegalNERService
from app.services.ocr.postprocess import LegalPostProcessor, LegalEntity

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extrator de entidades estruturadas em textos jurídicos.

    Combina NER (BERT-based) e RegEx para máxima cobertura.
    """

    def __init__(self) -> None:
        self.ner_service = LegalNERService()
        self.regex_processor = LegalPostProcessor()

    def get_case_parties(self, text: str) -> list[dict]:
        """
        Extrai partes do processo (autor, réu, advogados).

        Usa NER (PESSOA, ORGANIZACAO) + RegEx (OAB, CPF, CNPJ)
        para inferir qualificação das partes.

        Args:
            text: Texto do documento/peça processual

        Returns:
            Lista de dicionários com name, role, oab, cpf_cnpj
        """
        entities = self.ner_service.extract_entities(text)
        regex_entities = self.regex_processor.extract_entities(text)

        parties: list[dict] = []

        # Pessoas e organizações do NER
        for ent in entities:
            if ent.label in ("PESSOA", "ORGANIZACAO") and len(ent.text) > 3:
                parties.append(
                    {
                        "name": ent.text.strip(),
                        "role": None,
                        "oab": None,
                        "cpf_cnpj": None,
                    }
                )

        # Enriquecer com OAB/CPF/CNPJ quando próximos
        for ent in regex_entities:
            if ent.label == "OAB":
                if parties:
                    parties[-1]["oab"] = ent.text
            elif ent.label in ("CPF", "CNPJ"):
                if parties:
                    parties[-1]["cpf_cnpj"] = ent.text

        return parties

    def get_monetary_values(self, text: str) -> list[dict]:
        """
        Extrai valores monetários do texto.

        Reconhece R$ X.XXX,XX e formas por extenso (mil, milhões).

        Args:
            text: Texto a analisar

        Returns:
            Lista de {value: Decimal, raw_text: str}
        """
        entities = self.regex_processor.extract_entities(text)
        result: list[dict] = []

        for ent in entities:
            if ent.label != "MONETARY_VALUE":
                continue

            raw = ent.text
            try:
                # Normalizar: R$ 1.234,56 -> 1234.56
                num_str = raw.replace("R$", "").strip()
                num_str = num_str.replace(".", "").replace(",", ".")

                if "mil" in num_str.lower():
                    base = float(num_str.lower().split("mil")[0].strip() or "1")
                    value = Decimal(str(base * 1000))
                elif "milh" in num_str.lower():
                    base = float(
                        num_str.lower()
                        .replace("milhões", "")
                        .replace("milhão", "")
                        .strip()
                        or "1"
                    )
                    value = Decimal(str(base * 1_000_000))
                else:
                    value = Decimal(num_str)

                result.append({"value": value, "raw_text": raw})
            except Exception as e:
                logger.warning("Failed to parse monetary value %r: %s", raw, e)
                result.append({"value": None, "raw_text": raw})

        return result

    def get_dates(self, text: str) -> list[dict]:
        """
        Extrai datas do texto.

        Suporta: DD/MM/AAAA, DD de Mês de AAAA, etc.

        Args:
            text: Texto a analisar

        Returns:
            Lista de {date: datetime|str, raw_text: str}
        """
        import re

        entities = self.regex_processor.extract_entities(text)
        result: list[dict] = []

        # Data por extenso
        for ent in entities:
            if ent.label == "DATE_BR":
                result.append({"date": None, "raw_text": ent.text})

        # DD/MM/AAAA
        date_pattern = re.compile(
            r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b"
        )
        for match in date_pattern.finditer(text):
            d, m, a = match.groups()
            try:
                dt = datetime(int(a), int(m), int(d))
                result.append({"date": dt, "raw_text": match.group()})
            except ValueError:
                result.append({"date": None, "raw_text": match.group()})

        return result
