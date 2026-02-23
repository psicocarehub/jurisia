// Legal document types based on VICTOR/STF classification
export const DOCUMENT_TYPES = [
  "peticao_inicial",
  "contestacao",
  "replica",
  "sentenca",
  "acordao",
  "decisao_monocratica",
  "recurso_apelacao",
  "recurso_agravo",
  "recurso_extraordinario",
  "recurso_especial",
  "certidao",
  "procuracao",
  "comprovante",
  "laudo_parecer",
  "despacho",
  "outros",
] as const;

// Legal areas
export const LEGAL_AREAS = [
  "civil",
  "trabalhista",
  "criminal",
  "tributario",
  "administrativo",
  "consumidor",
  "familia",
  "empresarial",
  "ambiental",
  "previdenciario",
  "eleitoral",
  "constitucional",
] as const;

// Case statuses
export const CASE_STATUSES = [
  "active",
  "archived",
  "closed",
  "suspended",
] as const;

// Petition statuses
export const PETITION_STATUSES = [
  "draft",
  "review",
  "approved",
  "filed",
] as const;

// AI compliance labels
export const AI_DISCLAIMER =
  "⚠️ Conteúdo gerado com auxílio de inteligência artificial. " +
  "Conforme CNJ Resolução 615/2025 e recomendações da OAB, " +
  "este conteúdo deve ser revisado por advogado habilitado " +
  "antes de qualquer uso em processo judicial.";

export const AI_LABEL_SHORT =
  "Conteúdo gerado com auxílio de IA — CNJ Res. 615/2025";
