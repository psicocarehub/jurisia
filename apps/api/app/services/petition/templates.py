"""
Petition templates for Brazilian legal system.
16 types with ABNT structure, required sections, and placeholders.
"""

from typing import Optional


TEMPLATES: dict[str, dict] = {
    "peticao_inicial": {
        "id": "peticao_inicial",
        "name": "Petição Inicial",
        "category": "cível",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True,
             "placeholder": "EXCELENTÍSSIMO(A) SENHOR(A) DOUTOR(A) JUIZ(A) DE DIREITO DA __ VARA {area} DA COMARCA DE {comarca}/{uf}"},
            {"id": "qualificacao_autor", "title": "Qualificação do Autor", "required": True,
             "placeholder": "{nome_autor}, {nacionalidade}, {estado_civil}, {profissao}, inscrito(a) no CPF sob nº {cpf}, RG nº {rg}, residente e domiciliado(a) em {endereco_autor}"},
            {"id": "qualificacao_reu", "title": "Qualificação do Réu", "required": True,
             "placeholder": "em face de {nome_reu}, {qualificacao_reu}, residente e domiciliado(a) em {endereco_reu}"},
            {"id": "fatos", "title": "DOS FATOS", "required": True,
             "placeholder": "Narrar os fatos que ensejam a demanda, em ordem cronológica."},
            {"id": "direito", "title": "DO DIREITO", "required": True,
             "placeholder": "Fundamentação jurídica com artigos de lei, doutrina e jurisprudência."},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True,
             "placeholder": "Ante o exposto, requer:\na) ...\nb) ...\nc) Condenação do réu ao pagamento das custas processuais e honorários advocatícios."},
            {"id": "valor_causa", "title": "DO VALOR DA CAUSA", "required": True,
             "placeholder": "Dá-se à causa o valor de R$ {valor_causa} ({valor_extenso})."},
            {"id": "provas", "title": "DAS PROVAS", "required": False,
             "placeholder": "Protesta pela produção de todas as provas em direito admitidas."},
            {"id": "encerramento", "title": "Encerramento", "required": True,
             "placeholder": "Nestes termos,\npede deferimento.\n\n{cidade}, {data}.\n\n{advogado}\nOAB/{uf} nº {oab}"},
        ],
    },
    "contestacao": {
        "id": "contestacao",
        "name": "Contestação",
        "category": "cível",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True,
             "placeholder": "EXCELENTÍSSIMO(A) SENHOR(A) DOUTOR(A) JUIZ(A) DE DIREITO DA __ VARA {area} DA COMARCA DE {comarca}/{uf}\n\nProcesso nº {numero_processo}"},
            {"id": "qualificacao", "title": "Qualificação do Contestante", "required": True,
             "placeholder": "{nome_reu}, já qualificado(a) nos autos do processo em epígrafe"},
            {"id": "preliminares", "title": "DAS PRELIMINARES", "required": False,
             "placeholder": "I - Da inépcia da inicial\nII - Da ilegitimidade passiva\nIII - Da prescrição"},
            {"id": "merito", "title": "DO MÉRITO", "required": True,
             "placeholder": "Impugnação específica dos fatos narrados na inicial."},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True,
             "placeholder": "Ante o exposto, requer:\na) O acolhimento das preliminares;\nb) No mérito, a total improcedência dos pedidos;\nc) Condenação do autor em custas e honorários."},
            {"id": "provas", "title": "DAS PROVAS", "required": False,
             "placeholder": "Protesta pela produção de todas as provas em direito admitidas."},
            {"id": "encerramento", "title": "Encerramento", "required": True,
             "placeholder": "Nestes termos,\npede deferimento."},
        ],
    },
    "recurso_apelacao": {
        "id": "recurso_apelacao",
        "name": "Recurso de Apelação",
        "category": "recursal",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True,
             "placeholder": "EXCELENTÍSSIMO(A) SENHOR(A) DOUTOR(A) JUIZ(A) DE DIREITO DA __ VARA {area}"},
            {"id": "interposicao", "title": "Interposição", "required": True,
             "placeholder": "{nome_apelante}, nos autos da ação {tipo_acao} nº {numero_processo}, inconformado(a) com a r. sentença de fls., vem interpor RECURSO DE APELAÇÃO"},
            {"id": "razoes", "title": "RAZÕES DE APELAÇÃO", "required": True,
             "placeholder": "Endereçado ao Tribunal de Justiça do Estado de {uf}\n\nI - DOS FATOS\nII - DO DIREITO\nIII - DO PEDIDO DE REFORMA"},
            {"id": "preliminar_recursal", "title": "DAS PRELIMINARES RECURSAIS", "required": False,
             "placeholder": "Da tempestividade, do preparo, etc."},
            {"id": "merito_recursal", "title": "DO MÉRITO RECURSAL", "required": True,
             "placeholder": "Razões pelas quais a sentença merece reforma."},
            {"id": "pedido_reforma", "title": "DO PEDIDO", "required": True,
             "placeholder": "Ante o exposto, requer o conhecimento e provimento do recurso para reformar a r. sentença."},
            {"id": "encerramento", "title": "Encerramento", "required": True,
             "placeholder": "Nestes termos,\npede deferimento."},
        ],
    },
    "habeas_corpus": {
        "id": "habeas_corpus",
        "name": "Habeas Corpus",
        "category": "criminal",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True,
             "placeholder": "EXCELENTÍSSIMO(A) SENHOR(A) DOUTOR(A) DESEMBARGADOR(A) PRESIDENTE DO TRIBUNAL DE JUSTIÇA DO ESTADO DE {uf}"},
            {"id": "impetrante", "title": "DO IMPETRANTE", "required": True,
             "placeholder": "{nome_impetrante}, advogado(a), inscrito(a) na OAB/{uf} sob o nº {oab}"},
            {"id": "paciente", "title": "DO PACIENTE", "required": True,
             "placeholder": "em favor de {nome_paciente}, {qualificacao_paciente}"},
            {"id": "autoridade_coatora", "title": "DA AUTORIDADE COATORA", "required": True,
             "placeholder": "Juízo da __ Vara Criminal da Comarca de {comarca}"},
            {"id": "fatos", "title": "DOS FATOS", "required": True,
             "placeholder": "Narração dos fatos que configuram a coação ilegal."},
            {"id": "direito", "title": "DO DIREITO", "required": True,
             "placeholder": "Art. 5º, LXVIII da CF/88. Art. 647 e ss. do CPP."},
            {"id": "liminar", "title": "DA LIMINAR", "required": False,
             "placeholder": "Requer a concessão de medida liminar para cessar a coação."},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True,
             "placeholder": "a) Concessão de liminar;\nb) No mérito, concessão definitiva da ordem."},
            {"id": "encerramento", "title": "Encerramento", "required": True,
             "placeholder": "Nestes termos,\npede deferimento."},
        ],
    },
    "mandado_seguranca": {
        "id": "mandado_seguranca",
        "name": "Mandado de Segurança",
        "category": "constitucional",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True,
             "placeholder": "EXCELENTÍSSIMO(A) SENHOR(A) DOUTOR(A) JUIZ(A) FEDERAL DA __ VARA DA SEÇÃO JUDICIÁRIA DE {uf}"},
            {"id": "impetrante", "title": "DO IMPETRANTE", "required": True, "placeholder": ""},
            {"id": "autoridade_coatora", "title": "DA AUTORIDADE COATORA", "required": True, "placeholder": ""},
            {"id": "fatos", "title": "DOS FATOS", "required": True, "placeholder": ""},
            {"id": "direito_liquido_certo", "title": "DO DIREITO LÍQUIDO E CERTO", "required": True,
             "placeholder": "Art. 5º, LXIX da CF/88. Lei 12.016/2009."},
            {"id": "liminar", "title": "DA LIMINAR", "required": True, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "agravo_instrumento": {
        "id": "agravo_instrumento",
        "name": "Agravo de Instrumento",
        "category": "recursal",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True, "placeholder": ""},
            {"id": "interposicao", "title": "Interposição", "required": True, "placeholder": ""},
            {"id": "decisao_agravada", "title": "DA DECISÃO AGRAVADA", "required": True, "placeholder": ""},
            {"id": "razoes", "title": "DAS RAZÕES", "required": True, "placeholder": ""},
            {"id": "efeito_suspensivo", "title": "DO EFEITO SUSPENSIVO", "required": False, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "recurso_especial": {
        "id": "recurso_especial",
        "name": "Recurso Especial (REsp)",
        "category": "recursal",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True,
             "placeholder": "EXCELENTÍSSIMO(A) SENHOR(A) MINISTRO(A) PRESIDENTE DO SUPERIOR TRIBUNAL DE JUSTIÇA"},
            {"id": "cabimento", "title": "DO CABIMENTO", "required": True,
             "placeholder": "Art. 105, III, da CF/88. Prequestionamento."},
            {"id": "razoes", "title": "DAS RAZÕES DO RECURSO", "required": True, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "recurso_extraordinario": {
        "id": "recurso_extraordinario",
        "name": "Recurso Extraordinário (RE)",
        "category": "recursal",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True,
             "placeholder": "EXCELENTÍSSIMO(A) SENHOR(A) MINISTRO(A) PRESIDENTE DO SUPREMO TRIBUNAL FEDERAL"},
            {"id": "repercussao_geral", "title": "DA REPERCUSSÃO GERAL", "required": True,
             "placeholder": "Art. 102, III e §3º da CF/88."},
            {"id": "razoes", "title": "DAS RAZÕES DO RECURSO", "required": True, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "embargos_declaracao": {
        "id": "embargos_declaracao",
        "name": "Embargos de Declaração",
        "category": "recursal",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True, "placeholder": ""},
            {"id": "tempestividade", "title": "DA TEMPESTIVIDADE", "required": True, "placeholder": "Art. 1.023 do CPC. Prazo de 5 dias."},
            {"id": "vicio", "title": "DO VÍCIO", "required": True,
             "placeholder": "Obscuridade / Contradição / Omissão (Art. 1.022 do CPC)"},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "reclamacao_trabalhista": {
        "id": "reclamacao_trabalhista",
        "name": "Reclamação Trabalhista",
        "category": "trabalhista",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True,
             "placeholder": "EXCELENTÍSSIMO(A) SENHOR(A) DOUTOR(A) JUIZ(A) DO TRABALHO DA __ VARA DO TRABALHO DE {comarca}/{uf}"},
            {"id": "qualificacao_reclamante", "title": "DO RECLAMANTE", "required": True, "placeholder": ""},
            {"id": "qualificacao_reclamada", "title": "DA RECLAMADA", "required": True, "placeholder": ""},
            {"id": "contrato", "title": "DO CONTRATO DE TRABALHO", "required": True, "placeholder": ""},
            {"id": "verbas", "title": "DAS VERBAS RESCISÓRIAS", "required": True, "placeholder": ""},
            {"id": "fgts", "title": "DO FGTS", "required": False, "placeholder": ""},
            {"id": "horas_extras", "title": "DAS HORAS EXTRAS", "required": False, "placeholder": ""},
            {"id": "danos_morais", "title": "DOS DANOS MORAIS", "required": False, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "valor_causa", "title": "DO VALOR DA CAUSA", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "execucao_titulo": {
        "id": "execucao_titulo",
        "name": "Execução de Título Extrajudicial",
        "category": "cível",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True, "placeholder": ""},
            {"id": "qualificacao", "title": "DAS PARTES", "required": True, "placeholder": ""},
            {"id": "titulo", "title": "DO TÍTULO EXECUTIVO", "required": True, "placeholder": ""},
            {"id": "debito", "title": "DO DÉBITO", "required": True, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True,
             "placeholder": "a) Citação do executado para pagar em 3 dias (Art. 829 CPC);\nb) Penhora de bens."},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "acao_alimentos": {
        "id": "acao_alimentos",
        "name": "Ação de Alimentos",
        "category": "família",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True, "placeholder": ""},
            {"id": "qualificacao", "title": "DAS PARTES", "required": True, "placeholder": ""},
            {"id": "fatos", "title": "DOS FATOS", "required": True, "placeholder": ""},
            {"id": "necessidade", "title": "DA NECESSIDADE DO ALIMENTANDO", "required": True, "placeholder": ""},
            {"id": "possibilidade", "title": "DA POSSIBILIDADE DO ALIMENTANTE", "required": True, "placeholder": ""},
            {"id": "alimentos_provisorios", "title": "DOS ALIMENTOS PROVISÓRIOS", "required": True,
             "placeholder": "Art. 4º da Lei 5.478/68."},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "tutela_antecipada": {
        "id": "tutela_antecipada",
        "name": "Tutela Antecipada Antecedente",
        "category": "cível",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True, "placeholder": ""},
            {"id": "qualificacao", "title": "DAS PARTES", "required": True, "placeholder": ""},
            {"id": "fatos", "title": "DOS FATOS", "required": True, "placeholder": ""},
            {"id": "urgencia", "title": "DA URGÊNCIA E PROBABILIDADE DO DIREITO", "required": True,
             "placeholder": "Art. 300 e 303 do CPC."},
            {"id": "perigo_dano", "title": "DO PERIGO DE DANO", "required": True, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "replica": {
        "id": "replica",
        "name": "Réplica à Contestação",
        "category": "cível",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True, "placeholder": ""},
            {"id": "preliminares", "title": "DAS PRELIMINARES", "required": False,
             "placeholder": "Refutação das preliminares arguidas na contestação."},
            {"id": "merito", "title": "DO MÉRITO", "required": True,
             "placeholder": "Contrarrazões aos argumentos de mérito da contestação."},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True,
             "placeholder": "Reitera os pedidos da inicial."},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "impugnacao_cumprimento": {
        "id": "impugnacao_cumprimento",
        "name": "Impugnação ao Cumprimento de Sentença",
        "category": "cível",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True, "placeholder": ""},
            {"id": "tempestividade", "title": "DA TEMPESTIVIDADE", "required": True,
             "placeholder": "Art. 525 do CPC. Prazo de 15 dias."},
            {"id": "materias", "title": "DAS MATÉRIAS ARGUÍVEIS", "required": True,
             "placeholder": "Art. 525, §1º do CPC."},
            {"id": "excesso_execucao", "title": "DO EXCESSO DE EXECUÇÃO", "required": False, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True, "placeholder": ""},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
    "denuncia_crime": {
        "id": "denuncia_crime",
        "name": "Queixa-Crime",
        "category": "criminal",
        "sections": [
            {"id": "enderecamento", "title": "Endereçamento", "required": True, "placeholder": ""},
            {"id": "querelante", "title": "DO QUERELANTE", "required": True, "placeholder": ""},
            {"id": "querelado", "title": "DO QUERELADO", "required": True, "placeholder": ""},
            {"id": "fatos", "title": "DOS FATOS", "required": True, "placeholder": ""},
            {"id": "tipificacao", "title": "DA TIPIFICAÇÃO", "required": True, "placeholder": ""},
            {"id": "pedidos", "title": "DOS PEDIDOS", "required": True,
             "placeholder": "a) Recebimento da queixa-crime;\nb) Citação do querelado;\nc) Condenação nas penas do art. __ do CP."},
            {"id": "encerramento", "title": "Encerramento", "required": True, "placeholder": ""},
        ],
    },
}


def get_template(template_id: str) -> Optional[dict]:
    """Get template by ID."""
    return TEMPLATES.get(template_id)


def list_templates(petition_type: Optional[str] = None, category: Optional[str] = None) -> list[dict]:
    """List available templates, optionally filtered by category."""
    templates = list(TEMPLATES.values())
    if petition_type:
        templates = [t for t in templates if t["id"] == petition_type]
    if category:
        templates = [t for t in templates if t.get("category") == category]
    return [
        {"id": t["id"], "name": t["name"], "category": t.get("category", ""), "sections_count": len(t["sections"])}
        for t in templates
    ]


def render_template(template_id: str, variables: dict[str, str]) -> str:
    """Render a template with variable substitution, producing the skeleton text."""
    template = TEMPLATES.get(template_id)
    if not template:
        return ""

    parts = []
    for section in template["sections"]:
        title = section["title"]
        placeholder = section.get("placeholder", "")
        text = placeholder
        for key, value in variables.items():
            text = text.replace(f"{{{key}}}", value)
        if title.startswith("D") or title.startswith("Encerramento"):
            parts.append(f"\n{title}\n\n{text}")
        else:
            parts.append(f"{text}")

    return "\n\n".join(parts)
