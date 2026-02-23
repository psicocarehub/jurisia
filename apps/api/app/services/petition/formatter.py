"""
Petition formatter -- formatacao ABNT, OAB e labels CNJ.

Aplica estilos HTML inline conforme ABNT NBR 6023/6024,
OAB (qualificacao, enderecamento) e disclaimer CNJ Res. 615/2025.
"""

import re
from typing import Optional

AI_DISCLAIMER = (
    "Conteudo gerado com auxilio de inteligencia artificial. "
    "Conforme CNJ Resolucao 615/2025 e recomendacoes da OAB "
    "(Proposicao 49.0000.2024.007325-9/COP), este conteudo deve ser "
    "revisado por advogado habilitado antes de qualquer uso em processo judicial."
)

ABNT_STYLES = {
    "body": (
        "font-family: 'Times New Roman', Times, serif; "
        "font-size: 12pt; "
        "line-height: 1.5; "
        "color: #000; "
        "text-align: justify;"
    ),
    "margins": (
        "margin-left: 3cm; "
        "margin-right: 2cm; "
        "margin-top: 3cm; "
        "margin-bottom: 2cm;"
    ),
    "heading": (
        "font-family: 'Times New Roman', Times, serif; "
        "font-size: 14pt; "
        "font-weight: bold; "
        "text-align: center; "
        "text-transform: uppercase; "
        "margin-bottom: 24pt;"
    ),
    "paragraph": (
        "text-indent: 1.25cm; "
        "margin-bottom: 6pt; "
        "text-align: justify;"
    ),
    "citation_block": (
        "font-size: 10pt; "
        "line-height: 1.0; "
        "margin-left: 4cm; "
        "margin-top: 12pt; "
        "margin-bottom: 12pt; "
        "text-align: justify;"
    ),
    "footer": (
        "font-size: 10pt; "
        "text-align: center; "
        "border-top: 1px solid #ccc; "
        "padding-top: 8pt; "
        "margin-top: 24pt; "
        "color: #666;"
    ),
}


class PetitionFormatter:
    """Formatador de peticoes juridicas com suporte a ABNT, OAB e CNJ."""

    def format_abnt(
        self,
        text: str,
        font: str = "Times New Roman",
        size: int = 12,
        spacing: float = 1.5,
        margin_left_cm: float = 3.0,
        margin_right_cm: float = 2.0,
    ) -> str:
        """
        Formata HTML conforme padroes ABNT.

        Aplica: fonte Times New Roman 12pt, espacamento 1.5,
        margens 3cm/2cm, recuo de paragrafo 1.25cm,
        citacoes longas em bloco recuado com fonte 10pt.
        """
        body_style = (
            f"font-family: '{font}', Times, serif; "
            f"font-size: {size}pt; "
            f"line-height: {spacing}; "
            f"color: #000; "
            f"text-align: justify; "
            f"margin-left: {margin_left_cm}cm; "
            f"margin-right: {margin_right_cm}cm; "
            f"margin-top: 3cm; "
            f"margin-bottom: 2cm;"
        )

        paragraphs = re.split(r'\n\s*\n|\<br\s*/?\>\s*\<br\s*/?\>', text)
        formatted_parts = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if self._is_heading(para):
                formatted_parts.append(
                    f'<h2 style="{ABNT_STYLES["heading"]}">{self._strip_html(para)}</h2>'
                )
            elif self._is_long_citation(para):
                formatted_parts.append(
                    f'<blockquote style="{ABNT_STYLES["citation_block"]}">{para}</blockquote>'
                )
            elif para.startswith('<'):
                formatted_parts.append(para)
            else:
                formatted_parts.append(
                    f'<p style="{ABNT_STYLES["paragraph"]}">{para}</p>'
                )

        content = '\n'.join(formatted_parts)
        return f'<div style="{body_style}">{content}</div>'

    def format_oab(
        self,
        text: str,
        include_header: bool = True,
    ) -> str:
        """
        Formata conforme padroes OAB.

        Inclui estrutura de enderecamento e qualificacao
        esperada pelas normas da OAB para pecas processuais.
        """
        parts = []

        if include_header:
            header = (
                '<div style="text-align: center; font-weight: bold; '
                'font-size: 14pt; margin-bottom: 24pt; text-transform: uppercase;">'
                'EXCELENTISSIMO(A) SENHOR(A) DOUTOR(A) JUIZ(A) DE DIREITO</div>'
            )
            parts.append(header)

        formatted = self.format_abnt(text)
        parts.append(formatted)

        location_date = (
            '<div style="text-align: right; margin-top: 36pt; font-style: italic;">'
            '[Local], [data].</div>'
        )
        parts.append(location_date)

        signature = (
            '<div style="text-align: center; margin-top: 48pt;">'
            '<div style="border-top: 1px solid #000; width: 250px; margin: 0 auto; padding-top: 8pt;">'
            '[Nome do Advogado]<br/>OAB/[UF] nº [número]</div></div>'
        )
        parts.append(signature)

        return '\n'.join(parts)

    def add_ai_label(
        self,
        text: str,
        position: str = "footer",
        custom_disclaimer: Optional[str] = None,
    ) -> str:
        """
        Injeta disclaimer CNJ Res. 615/2025 no conteudo.

        Obrigatorio para todo conteudo gerado por IA.
        """
        disclaimer = custom_disclaimer or AI_DISCLAIMER
        disclaimer_html = (
            f'<div style="{ABNT_STYLES["footer"]}">'
            f'<strong>Aviso:</strong> {disclaimer}</div>'
        )

        if position == "header":
            return disclaimer_html + text.strip()
        return text.strip() + disclaimer_html

    @staticmethod
    def _is_heading(text: str) -> bool:
        clean = re.sub(r'<[^>]+>', '', text).strip()
        heading_patterns = [
            r'^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)[\s.–-]',
            r'^\d+[\s.–-]\s*[A-Z]',
            r'^(?:DOS? |DAS? |DO |DA |PRELIMINARMENTE|MERITO|PEDIDOS?|CONCLUS)',
        ]
        if len(clean) < 80 and clean == clean.upper() and len(clean) > 3:
            return True
        return any(re.match(p, clean) for p in heading_patterns)

    @staticmethod
    def _is_long_citation(text: str) -> bool:
        clean = re.sub(r'<[^>]+>', '', text).strip()
        if len(clean) > 200 and (clean.startswith('"') or clean.startswith("'")):
            return True
        if re.match(r'^".*"$', clean, re.DOTALL) and len(clean) > 200:
            return True
        return False

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r'<[^>]+>', '', text).strip()
