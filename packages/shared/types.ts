// Shared types between frontend and backend

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  case_id?: string;
  session_id?: string;
  stream?: boolean;
  use_rag?: boolean;
  use_memory?: boolean;
}

export interface ChatResponse {
  message: ChatMessage;
  sources: Source[];
  thinking?: string;
  model_used: string;
}

export interface Source {
  id: string;
  content: string;
  score: number;
  document_title: string;
  doc_type: string;
  court: string;
  date: string;
}

export interface Case {
  id: string;
  title: string;
  cnj_number?: string;
  description?: string;
  area?: string;
  status: string;
  client_name?: string;
  opposing_party?: string;
  court?: string;
  judge_name?: string;
  created_at?: string;
}

export interface Document {
  id: string;
  title: string;
  doc_type?: string;
  source?: string;
  ocr_status: string;
  classification_label?: string;
  created_at?: string;
}

export interface Petition {
  id: string;
  title: string;
  petition_type?: string;
  status: string;
  ai_generated: boolean;
  ai_label: string;
}

export interface JudgeProfile {
  name: string;
  court?: string;
  total_decisions: number;
  avg_decision_time_days?: number;
  favorability: Record<string, Record<string, number>>;
}

export type CitationStatus =
  | "verified"
  | "not_found"
  | "revoked"
  | "outdated"
  | "unchecked";

export interface Citation {
  text: string;
  type: "legislacao" | "jurisprudencia" | "doutrina";
  status: CitationStatus;
  verified_text?: string;
  source_url?: string;
  confidence: number;
}
