"""
Petition formatter — formatação ABNT, OAB e labels CNJ.

Formatação de petições conforme padrões técnicos e
injeção de disclaimer CNJ Res. 615/2025.
"""

from typing import Optional

# CNJ Res. 615/2025 — obrigatório em conteúdo gerado por IA
AI_DISCLAIMER = (
    "⚠️ Conteúdo gerado com auxílio de inteligência artificial. "
    "Conforme CNJ Resolução 615/2025 e recomendações da OAB "
    "(Proposição 49.0000.2024.007325-9/COP), este conteúdo deve ser "
    "revisado por advogado habilitado antes de qualquer uso em processo judicial."
)


class PetitionFormatter:
    """
    Formatador de petições jurídicas.

    Suporta ABNT, padrões OAB e inserção de disclaimer de IA.
    """

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
        Formata texto conforme padrões ABNT.

        Args:
            text: Conteúdo da petição
            font: Fonte (Times New Roman padrão ABNT)
            size: Tamanho em pontos
            spacing: Espaçamento entre linhas
            margin_left_cm: Margem esquerda em cm
            margin_right_cm: Margem direita em cm

        Returns:
            Texto com instruções de formatação ABNT (stub: retorna texto)
        """
        # Stub: em produção, gerar HTML/Markdown com metadados ABNT
        _ = font, size, spacing, margin_left_cm, margin_right_cm
        return text.strip()

    def format_oab(
        self,
        text: str,
        include_header: bool = True,
    ) -> str:
        """
        Formata conforme padrões OAB.

        Inclui qualificação, endereçamento e estrutura esperada
        pelas normas da OAB para peças processuais.

        Args:
            text: Conteúdo da petição
            include_header: Incluir cabeçalho padrão OAB

        Returns:
            Texto formatado (stub: retorna texto)
        """
        _ = include_header
        return text.strip()

    def add_ai_label(
        self,
        text: str,
        position: str = "footer",
        custom_disclaimer: Optional[str] = None,
    ) -> str:
        """
        Injeta disclaimer CNJ Res. 615/2025 no conteúdo.

        Obrigatório para todo conteúdo gerado por IA conforme
        CNJ Resolução 615/2025 e OAB Item 3.7.

        Args:
            text: Conteúdo da petição
            position: 'header' ou 'footer' — onde inserir o aviso
            custom_disclaimer: Texto customizado (usa default se None)

        Returns:
            Texto com disclaimer adicionado
        """
        disclaimer = custom_disclaimer or AI_DISCLAIMER
        block = f"\n\n{disclaimer}\n\n"

        if position == "header":
            return block + text.strip()
        return text.strip() + block
