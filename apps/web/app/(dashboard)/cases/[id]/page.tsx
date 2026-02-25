'use client';

import { useEffect, useState, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { apiFetch, apiPost } from '@/lib/api';
import { useToast } from '@/components/toast';
import {
  ArrowLeft,
  Scan,
  FileText,
  Shield,
  Lightbulb,
  User,
  TrendingUp,
  Scale,
  AlertTriangle,
  Clock,
  ChevronDown,
  ChevronUp,
  Loader2,
  BookOpen,
} from 'lucide-react';

interface CaseData {
  id: string;
  title: string;
  cnj_number?: string;
  description?: string;
  area?: string;
  status: string;
  client_name?: string;
  client_document?: string;
  opposing_party?: string;
  court?: string;
  judge_name?: string;
  estimated_value?: number;
  created_at?: string;
}

interface Vulnerability {
  type: string;
  title: string;
  description: string;
  severity: string;
  legal_basis: string;
  recommendation: string;
}

interface Strategy {
  type: string;
  title: string;
  description: string;
  legal_basis: string;
  success_likelihood: string;
  risks: string;
}

interface TimelineEvent {
  date: string;
  event: string;
}

interface Analysis {
  case_id: string;
  summary: string;
  timeline: TimelineEvent[];
  legal_framework: string;
  vulnerabilities: Vulnerability[];
  strategies: Strategy[];
  judge_profile: Record<string, unknown> | null;
  prediction: Record<string, unknown> | null;
  similar_cases: Array<{
    title: string;
    court: string;
    date: string;
    doc_type: string;
    snippet: string;
  }>;
  risk_level: string;
  risk_assessment: string;
  model_used: string;
  generated_at: string;
}

const SEVERITY_STYLES: Record<string, string> = {
  alta: 'bg-red-100 text-red-800',
  média: 'bg-yellow-100 text-yellow-800',
  baixa: 'bg-green-100 text-green-800',
};

const RISK_STYLES: Record<string, string> = {
  alto: 'bg-red-100 text-red-800 border-red-300',
  médio: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  baixo: 'bg-green-100 text-green-800 border-green-300',
};

const LIKELIHOOD_STYLES: Record<string, string> = {
  alta: 'text-green-700',
  média: 'text-yellow-700',
  baixa: 'text-red-700',
};

export default function CaseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const caseId = params.id as string;
  const { error: showError } = useToast();

  const [caseData, setCaseData] = useState<CaseData | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState<{ step: string; progress: number } | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    summary: true,
    vulnerabilities: true,
    strategies: true,
  });

  const fetchCase = useCallback(async () => {
    try {
      const res = await apiFetch(`/api/v1/cases/${caseId}`);
      if (res.ok) setCaseData(await res.json());
    } catch {
      showError('Erro ao carregar caso');
    }
  }, [caseId]);

  const fetchAnalysis = useCallback(async () => {
    try {
      const res = await apiFetch(`/api/v1/cases/${caseId}/analysis`);
      if (res.ok) setAnalysis(await res.json());
    } catch {
      // No cached analysis — that's fine
    }
  }, [caseId]);

  useEffect(() => {
    Promise.all([fetchCase(), fetchAnalysis()]).finally(() => setLoading(false));
  }, [fetchCase, fetchAnalysis]);

  const STEP_LABELS: Record<string, string> = {
    fetching_case: 'Carregando processo...',
    fetching_documents: 'Buscando documentos...',
    searching_jurisprudence: 'Pesquisando jurisprudência...',
    judge_profile: 'Analisando perfil do juiz...',
    outcome_prediction: 'Calculando predição...',
    llm_analysis: 'Executando análise com IA...',
    finalizing: 'Finalizando...',
    complete: 'Análise concluída!',
    error: 'Erro na análise',
  };

  const runAnalysis = async () => {
    setAnalyzing(true);
    setAnalysisProgress({ step: 'fetching_case', progress: 0 });

    try {
      const token = typeof window !== 'undefined' ? localStorage.getItem('jurisai_token') : null;
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || '';
      const url = `${baseUrl}/api/v1/cases/${caseId}/analyze/stream`;

      const res = await fetch(url, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Accept': 'text/event-stream',
        },
      });

      if (!res.ok || !res.body) {
        const fallback = await apiPost<Analysis>(`/api/v1/cases/${caseId}/analyze`, {});
        setAnalysis(fallback);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const match = line.match(/^data: (.+)$/m);
          if (!match) continue;
          try {
            const event = JSON.parse(match[1]);
            setAnalysisProgress({ step: event.step, progress: event.progress });

            if (event.step === 'complete' && event.data) {
              setAnalysis(event.data as Analysis);
            } else if (event.step === 'error') {
              showError(event.data?.message || 'Erro na análise');
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Erro ao analisar caso');
    } finally {
      setAnalyzing(false);
      setAnalysisProgress(null);
    }
  };

  const toggle = (section: string) =>
    setExpanded((prev) => ({ ...prev, [section]: !prev[section] }));

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-legal-blue-600" />
      </div>
    );
  }

  if (!caseData) {
    return (
      <div className="p-8 text-center text-gray-500">Caso não encontrado.</div>
    );
  }

  return (
    <div className="h-full overflow-auto bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => router.push('/cases')}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="h-5 w-5 text-gray-600" />
              </button>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  {caseData.title}
                </h1>
                <div className="flex items-center gap-3 mt-0.5 text-sm text-gray-500">
                  {caseData.cnj_number && <span>{caseData.cnj_number}</span>}
                  {caseData.area && (
                    <span className="px-2 py-0.5 bg-legal-blue-50 text-legal-blue-700 rounded text-xs">
                      {caseData.area}
                    </span>
                  )}
                  <span
                    className={`px-2 py-0.5 rounded text-xs ${
                      caseData.status === 'active'
                        ? 'bg-green-50 text-green-700'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {caseData.status === 'active' ? 'Ativo' : caseData.status}
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={runAnalysis}
              disabled={analyzing}
              className="btn-primary flex items-center gap-2 text-sm"
            >
              {analyzing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analisando...
                </>
              ) : (
                <>
                  <Scan className="h-4 w-4" />
                  {analysis ? 'Reanalisar' : 'Raio-X do Processo'}
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-6 space-y-6">
        {/* Case metadata */}
        <div className="bg-white rounded-xl border p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              ['Tribunal/Vara', caseData.court],
              ['Juiz', caseData.judge_name],
              ['Cliente', caseData.client_name],
              ['Parte Contrária', caseData.opposing_party],
              [
                'Valor da Causa',
                caseData.estimated_value
                  ? `R$ ${caseData.estimated_value.toLocaleString('pt-BR')}`
                  : null,
              ],
              ['Criado em', caseData.created_at?.split('T')[0]],
            ].map(([label, value]) =>
              value ? (
                <div key={label as string}>
                  <dt className="text-xs font-medium text-gray-500 uppercase">
                    {label}
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900">{value}</dd>
                </div>
              ) : null,
            )}
          </div>
          {caseData.description && (
            <div className="mt-4 pt-4 border-t">
              <p className="text-sm text-gray-700 whitespace-pre-wrap">
                {caseData.description}
              </p>
            </div>
          )}
        </div>

        {/* Analyzing progress */}
        {analyzing && !analysis && (
          <div className="space-y-4">
            {analysisProgress && (
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <div className="flex items-center gap-3 mb-3">
                  <Loader2 className="h-5 w-5 animate-spin text-legal-blue-600" />
                  <span className="text-sm font-medium text-gray-700">
                    {STEP_LABELS[analysisProgress.step] || analysisProgress.step}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-legal-blue-600 h-2.5 rounded-full transition-all duration-500"
                    style={{ width: `${analysisProgress.progress}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-2 text-right">{analysisProgress.progress}%</p>
              </div>
            )}
            <AnalysisSkeleton />
          </div>
        )}

        {/* Analysis results */}
        {analysis && (
          <>
            {/* Risk badge */}
            <div
              className={`rounded-xl border p-4 flex items-center gap-3 ${
                RISK_STYLES[analysis.risk_level] || RISK_STYLES['médio']
              }`}
            >
              <AlertTriangle className="h-5 w-5 shrink-0" />
              <div>
                <span className="font-semibold text-sm uppercase">
                  Risco {analysis.risk_level}
                </span>
                <p className="text-sm mt-0.5">{analysis.risk_assessment}</p>
              </div>
            </div>

            {/* Summary */}
            <CollapsibleSection
              title="Resumo dos Fatos"
              icon={<FileText className="h-5 w-5" />}
              expanded={expanded.summary}
              onToggle={() => toggle('summary')}
            >
              <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                {analysis.summary}
              </p>
            </CollapsibleSection>

            {/* Timeline */}
            {analysis.timeline.length > 0 && (
              <CollapsibleSection
                title="Timeline"
                icon={<Clock className="h-5 w-5" />}
                expanded={expanded.timeline}
                onToggle={() => toggle('timeline')}
              >
                <div className="space-y-3">
                  {analysis.timeline.map((evt, i) => (
                    <div key={i} className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <div className="w-2.5 h-2.5 rounded-full bg-legal-blue-500 mt-1.5" />
                        {i < analysis.timeline.length - 1 && (
                          <div className="w-0.5 flex-1 bg-gray-200 mt-1" />
                        )}
                      </div>
                      <div className="pb-4">
                        <span className="text-xs font-medium text-gray-500">
                          {evt.date}
                        </span>
                        <p className="text-sm text-gray-800">{evt.event}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Legal framework */}
            {analysis.legal_framework && (
              <CollapsibleSection
                title="Fundamentação Legal"
                icon={<BookOpen className="h-5 w-5" />}
                expanded={expanded.legal_framework}
                onToggle={() => toggle('legal_framework')}
              >
                <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                  {analysis.legal_framework}
                </p>
              </CollapsibleSection>
            )}

            {/* Vulnerabilities */}
            {analysis.vulnerabilities.length > 0 && (
              <CollapsibleSection
                title={`Brechas e Vulnerabilidades (${analysis.vulnerabilities.length})`}
                icon={<Shield className="h-5 w-5" />}
                expanded={expanded.vulnerabilities}
                onToggle={() => toggle('vulnerabilities')}
              >
                <div className="space-y-4">
                  {analysis.vulnerabilities.map((v, i) => (
                    <div
                      key={i}
                      className="border rounded-lg p-4 bg-gray-50"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="font-medium text-gray-900 text-sm">
                          {v.title}
                        </h4>
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-medium shrink-0 ${
                            SEVERITY_STYLES[v.severity] || 'bg-gray-100 text-gray-700'
                          }`}
                        >
                          {v.severity}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        {v.description}
                      </p>
                      {v.legal_basis && (
                        <p className="text-xs text-gray-500 mt-2">
                          <span className="font-medium">Base legal:</span>{' '}
                          {v.legal_basis}
                        </p>
                      )}
                      {v.recommendation && (
                        <p className="text-xs text-legal-blue-700 mt-1 bg-legal-blue-50 rounded p-2">
                          <span className="font-medium">Recomendação:</span>{' '}
                          {v.recommendation}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Strategies */}
            {analysis.strategies.length > 0 && (
              <CollapsibleSection
                title={`Estratégias Recomendadas (${analysis.strategies.length})`}
                icon={<Lightbulb className="h-5 w-5" />}
                expanded={expanded.strategies}
                onToggle={() => toggle('strategies')}
              >
                <div className="space-y-4">
                  {analysis.strategies.map((s, i) => (
                    <div
                      key={i}
                      className="border rounded-lg p-4 bg-gray-50"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="font-medium text-gray-900 text-sm">
                          {s.title}
                        </h4>
                        <div className="flex items-center gap-2 shrink-0">
                          <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">
                            {s.type}
                          </span>
                          {s.success_likelihood && (
                            <span
                              className={`text-xs font-medium ${
                                LIKELIHOOD_STYLES[s.success_likelihood] || ''
                              }`}
                            >
                              {s.success_likelihood}
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        {s.description}
                      </p>
                      {s.legal_basis && (
                        <p className="text-xs text-gray-500 mt-2">
                          <span className="font-medium">Fundamento:</span>{' '}
                          {s.legal_basis}
                        </p>
                      )}
                      {s.risks && (
                        <p className="text-xs text-orange-700 mt-1">
                          <span className="font-medium">Riscos:</span> {s.risks}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Judge profile */}
            {analysis.judge_profile && (
              <CollapsibleSection
                title="Perfil do Juiz"
                icon={<User className="h-5 w-5" />}
                expanded={expanded.judge}
                onToggle={() => toggle('judge')}
              >
                <JudgeProfileCard profile={analysis.judge_profile} />
              </CollapsibleSection>
            )}

            {/* Prediction */}
            {analysis.prediction && !analysis.prediction.warning && (
              <CollapsibleSection
                title="Predição de Resultado"
                icon={<TrendingUp className="h-5 w-5" />}
                expanded={expanded.prediction}
                onToggle={() => toggle('prediction')}
              >
                <PredictionCard prediction={analysis.prediction} />
              </CollapsibleSection>
            )}

            {/* Similar cases */}
            {analysis.similar_cases.length > 0 && (
              <CollapsibleSection
                title={`Jurisprudência Similar (${analysis.similar_cases.length})`}
                icon={<Scale className="h-5 w-5" />}
                expanded={expanded.similar}
                onToggle={() => toggle('similar')}
              >
                <div className="space-y-3">
                  {analysis.similar_cases.map((sc, i) => (
                    <div
                      key={i}
                      className="border rounded-lg p-3 bg-gray-50"
                    >
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        {sc.court && <span>{sc.court}</span>}
                        {sc.date && <span>{sc.date}</span>}
                        {sc.doc_type && (
                          <span className="px-1.5 py-0.5 bg-gray-200 rounded">
                            {sc.doc_type}
                          </span>
                        )}
                      </div>
                      <h4 className="text-sm font-medium text-gray-900 mt-1">
                        {sc.title || 'Sem título'}
                      </h4>
                      {sc.snippet && (
                        <p className="text-xs text-gray-600 mt-1 line-clamp-3">
                          {sc.snippet}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Footer */}
            <p className="text-xs text-gray-400 text-center pb-4">
              Análise gerada em{' '}
              {new Date(analysis.generated_at).toLocaleString('pt-BR')} |
              Modelo: {analysis.model_used} | Gerado com auxílio de IA (CNJ
              Res. 615/2025)
            </p>
          </>
        )}

        {/* No analysis yet */}
        {!analysis && !analyzing && (
          <div className="bg-white rounded-xl border p-12 text-center">
            <Scan className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-700">
              Raio-X do Processo
            </h3>
            <p className="text-sm text-gray-500 mt-2 max-w-md mx-auto">
              Clique em &quot;Raio-X do Processo&quot; para gerar uma análise
              completa: brechas, estratégias, perfil do juiz, predição e
              jurisprudência similar.
            </p>
            <button
              onClick={runAnalysis}
              className="btn-primary mt-6 inline-flex items-center gap-2"
            >
              <Scan className="h-4 w-4" />
              Analisar Agora
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function CollapsibleSection({
  title,
  icon,
  expanded,
  onToggle,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  expanded?: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  const isOpen = expanded ?? false;
  return (
    <div className="bg-white rounded-xl border">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-50 transition-colors rounded-xl"
      >
        <div className="flex items-center gap-2 text-gray-800 font-medium text-sm">
          {icon}
          {title}
        </div>
        {isOpen ? (
          <ChevronUp className="h-4 w-4 text-gray-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-gray-400" />
        )}
      </button>
      {isOpen && <div className="px-4 pb-4">{children}</div>}
    </div>
  );
}

function JudgeProfileCard({ profile }: { profile: Record<string, unknown> }) {
  const fav = (profile.favorability as Record<string, Record<string, number>>) || {};
  const geral = fav.geral || {};
  const topCitations = (profile.top_citations as Array<{ law: string; count: number }>) || [];
  const totalDecisions = (profile.total_decisions as number) || 0;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <dt className="text-xs text-gray-500">Nome</dt>
          <dd className="text-sm font-medium">{profile.name as string}</dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Tribunal</dt>
          <dd className="text-sm">{(profile.court as string) || 'N/A'}</dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Total Decisões</dt>
          <dd className="text-sm font-medium">{totalDecisions}</dd>
        </div>
        {geral.autor !== undefined && (
          <div>
            <dt className="text-xs text-gray-500">Favorabilidade</dt>
            <dd className="text-sm">
              Autor: <span className="font-medium text-green-700">{geral.autor}%</span>{' '}
              | Réu: <span className="font-medium text-red-700">{geral.reu}%</span>
            </dd>
          </div>
        )}
      </div>
      {topCitations.length > 0 && (
        <div>
          <dt className="text-xs text-gray-500 mb-1">Leis mais citadas</dt>
          <div className="flex flex-wrap gap-1.5">
            {topCitations.slice(0, 8).map((c, i) => (
              <span
                key={i}
                className="px-2 py-0.5 bg-gray-100 text-gray-700 rounded text-xs"
              >
                {c.law} ({c.count}x)
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PredictionCard({ prediction }: { prediction: Record<string, unknown> }) {
  const probs = (prediction.probabilities as Record<string, number>) || {};
  const outcome = prediction.outcome as string;
  const confidence = prediction.confidence as number;
  const factors = (prediction.factors as Array<{ name: string; value: string | number }>) || [];

  const labelMap: Record<string, string> = {
    procedente: 'Procedente',
    parcialmente_procedente: 'Parc. Procedente',
    improcedente: 'Improcedente',
    extinto_sem_merito: 'Extinto s/ Mérito',
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div>
          <span className="text-xs text-gray-500">Resultado provável</span>
          <p className="text-sm font-semibold text-gray-900">
            {labelMap[outcome] || outcome}
          </p>
        </div>
        <div>
          <span className="text-xs text-gray-500">Confiança</span>
          <p className="text-sm font-semibold">{(confidence * 100).toFixed(1)}%</p>
        </div>
      </div>
      {Object.keys(probs).length > 0 && (
        <div className="space-y-2">
          {Object.entries(probs).map(([key, val]) => (
            <div key={key} className="flex items-center gap-2">
              <span className="text-xs text-gray-600 w-32 shrink-0">
                {labelMap[key] || key}
              </span>
              <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-legal-blue-500 rounded-full transition-all"
                  style={{ width: `${val * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 w-12 text-right">
                {(val * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}
      {factors.length > 0 && (
        <div className="pt-2 border-t">
          <span className="text-xs text-gray-500">Fatores</span>
          <div className="grid grid-cols-2 gap-2 mt-1">
            {factors.map((f, i) => (
              <div key={i} className="text-xs text-gray-600">
                <span className="font-medium">{f.name}:</span> {String(f.value)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function AnalysisSkeleton() {
  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border p-6 text-center">
        <Loader2 className="h-8 w-8 animate-spin text-legal-blue-600 mx-auto mb-3" />
        <p className="text-sm font-medium text-gray-700">
          Analisando o processo...
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Buscando documentos, jurisprudência, perfil do juiz e gerando análise
          completa. Isso pode levar 15-30 segundos.
        </p>
      </div>
      {[1, 2, 3].map((i) => (
        <div key={i} className="bg-white rounded-xl border p-4 animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-3" />
          <div className="space-y-2">
            <div className="h-3 bg-gray-100 rounded w-full" />
            <div className="h-3 bg-gray-100 rounded w-5/6" />
            <div className="h-3 bg-gray-100 rounded w-4/6" />
          </div>
        </div>
      ))}
    </div>
  );
}
