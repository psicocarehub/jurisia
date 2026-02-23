'use client';

import { useState } from 'react';
import { apiFetch, apiPost } from '@/lib/api';
import {
  BarChart3,
  Search,
  User,
  TrendingUp,
  Scale,
  AlertTriangle,
} from 'lucide-react';

interface JudgeProfile {
  name: string;
  court: string;
  jurisdiction: string;
  total_decisions: number;
  favorability: Record<string, Record<string, number>>;
  top_citations: Array<{ law: string; count: number }>;
  decision_patterns: Record<string, number>;
  areas: Record<string, number>;
  recent_decisions: Array<{
    title: string;
    court: string;
    date: string;
    doc_type: string;
    snippet: string;
  }>;
}

interface PredictionResult {
  outcome: string;
  confidence: number;
  probabilities: Record<string, number>;
  factors: Array<{ name: string; value: string | number; importance?: number }>;
  warning?: string;
}

const AREAS = [
  'cível', 'trabalhista', 'tributário', 'família',
  'consumidor', 'administrativo', 'previdenciário', 'empresarial',
];

const TRIBUNAIS = [
  'STF', 'STJ', 'TST', 'TJSP', 'TJRJ', 'TJMG', 'TJRS',
  'TJPR', 'TJBA', 'TJSC', 'TRF1', 'TRF3', 'TRF4',
];

export default function JurimetricsPage() {
  const [activeTab, setActiveTab] = useState<'judge' | 'predict'>('judge');

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center gap-2 mb-6">
        <BarChart3 className="h-6 w-6 text-legal-blue-600" />
        <h1 className="text-xl font-semibold text-gray-900">Jurimetria</h1>
      </div>

      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveTab('judge')}
          className={`px-4 py-2 text-sm rounded-lg font-medium transition-colors ${
            activeTab === 'judge'
              ? 'bg-legal-blue-600 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          <User className="h-4 w-4 inline mr-1.5" />
          Perfil de Juiz
        </button>
        <button
          onClick={() => setActiveTab('predict')}
          className={`px-4 py-2 text-sm rounded-lg font-medium transition-colors ${
            activeTab === 'predict'
              ? 'bg-legal-blue-600 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          <Scale className="h-4 w-4 inline mr-1.5" />
          Predição de Resultado
        </button>
      </div>

      {activeTab === 'judge' ? <JudgeProfileTab /> : <PredictTab />}
    </div>
  );
}

function JudgeProfileTab() {
  const [judgeName, setJudgeName] = useState('');
  const [court, setCourt] = useState('');
  const [profile, setProfile] = useState<JudgeProfile | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!judgeName.trim()) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({ ...(court && { court }) });
      const res = await apiFetch(`/api/v1/jurimetrics/judges/${encodeURIComponent(judgeName)}?${params}`);
      if (res.ok) {
        setProfile(await res.json());
      }
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border p-4">
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Nome do juiz..."
              value={judgeName}
              onChange={(e) => setJudgeName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="input-field pl-9 text-sm"
            />
          </div>
          <select
            value={court}
            onChange={(e) => setCourt(e.target.value)}
            className="input-field text-sm w-32"
          >
            <option value="">Tribunal</option>
            {TRIBUNAIS.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <button onClick={handleSearch} disabled={loading} className="btn-primary text-sm">
            {loading ? 'Buscando...' : 'Buscar'}
          </button>
        </div>
      </div>

      {profile && (
        <div className="space-y-4">
          {/* Header */}
          <div className="bg-white rounded-xl border p-6">
            <div className="flex items-start gap-4">
              <div className="h-14 w-14 rounded-full bg-legal-blue-100 flex items-center justify-center">
                <User className="h-7 w-7 text-legal-blue-600" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">{profile.name}</h2>
                <p className="text-sm text-gray-500">{profile.court} {profile.jurisdiction && `| ${profile.jurisdiction}`}</p>
                <p className="text-sm text-gray-500 mt-1">{profile.total_decisions.toLocaleString()} decisões indexadas</p>
              </div>
            </div>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-3 gap-4">
            {/* Favorability */}
            {profile.favorability?.geral && (
              <div className="bg-white rounded-xl border p-4">
                <h3 className="text-xs font-medium text-gray-500 uppercase mb-3">Favorabilidade Geral</h3>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Autor</span>
                      <span className="font-medium text-green-600">{profile.favorability.geral.autor}%</span>
                    </div>
                    <div className="h-2 bg-gray-100 rounded-full">
                      <div className="h-2 bg-green-500 rounded-full" style={{ width: `${profile.favorability.geral.autor}%` }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Réu</span>
                      <span className="font-medium text-red-600">{profile.favorability.geral.reu}%</span>
                    </div>
                    <div className="h-2 bg-gray-100 rounded-full">
                      <div className="h-2 bg-red-500 rounded-full" style={{ width: `${profile.favorability.geral.reu}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Areas */}
            {Object.keys(profile.areas).length > 0 && (
              <div className="bg-white rounded-xl border p-4">
                <h3 className="text-xs font-medium text-gray-500 uppercase mb-3">Áreas de Atuação</h3>
                <div className="space-y-1.5">
                  {Object.entries(profile.areas).slice(0, 6).map(([area, count]) => (
                    <div key={area} className="flex justify-between text-sm">
                      <span className="text-gray-600 capitalize">{area}</span>
                      <span className="font-medium text-gray-900">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top Citations */}
            {profile.top_citations.length > 0 && (
              <div className="bg-white rounded-xl border p-4">
                <h3 className="text-xs font-medium text-gray-500 uppercase mb-3">Leis Mais Citadas</h3>
                <div className="space-y-1.5">
                  {profile.top_citations.slice(0, 6).map((c, i) => (
                    <div key={i} className="flex justify-between text-sm">
                      <span className="text-gray-600 truncate mr-2">{c.law}</span>
                      <span className="font-medium text-gray-900 shrink-0">{c.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Recent decisions */}
          {profile.recent_decisions.length > 0 && (
            <div className="bg-white rounded-xl border p-4">
              <h3 className="text-xs font-medium text-gray-500 uppercase mb-3">Decisões Recentes</h3>
              <div className="space-y-3">
                {profile.recent_decisions.map((d, i) => (
                  <div key={i} className="border-l-2 border-legal-blue-200 pl-3">
                    <h4 className="text-sm font-medium text-gray-900">{d.title}</h4>
                    <p className="text-xs text-gray-500">{d.court} | {d.date} | {d.doc_type}</p>
                    <p className="text-xs text-gray-600 mt-1">{d.snippet}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PredictTab() {
  const [form, setForm] = useState({
    area: 'cível',
    tribunal: 'TJSP',
    judge_name: '',
    estimated_value: '',
    tipo_acao: '',
    num_partes: '2',
  });
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const body = {
        ...form,
        estimated_value: form.estimated_value ? parseFloat(form.estimated_value) : undefined,
        num_partes: parseInt(form.num_partes) || 2,
        judge_name: form.judge_name || undefined,
      };
      const data = await apiPost<PredictionResult>('/api/v1/jurimetrics/predict', body);
      setResult(data);
    } catch (e: any) {
      alert(e.message || 'Erro na predição');
    } finally {
      setLoading(false);
    }
  };

  const outcomeLabel: Record<string, string> = {
    procedente: 'Procedente',
    parcialmente_procedente: 'Parcialmente Procedente',
    improcedente: 'Improcedente',
    extinto_sem_merito: 'Extinto sem Mérito',
  };

  const outcomeColor: Record<string, string> = {
    procedente: 'text-green-600',
    parcialmente_procedente: 'text-yellow-600',
    improcedente: 'text-red-600',
    extinto_sem_merito: 'text-gray-600',
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-sm font-medium text-gray-700 mb-4">Dados do Caso</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Área do Direito *</label>
            <select className="input-field text-sm" value={form.area} onChange={(e) => setForm({ ...form, area: e.target.value })}>
              {AREAS.map((a) => <option key={a} value={a}>{a.charAt(0).toUpperCase() + a.slice(1)}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Tribunal</label>
            <select className="input-field text-sm" value={form.tribunal} onChange={(e) => setForm({ ...form, tribunal: e.target.value })}>
              {TRIBUNAIS.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Juiz (opcional)</label>
            <input className="input-field text-sm" value={form.judge_name} onChange={(e) => setForm({ ...form, judge_name: e.target.value })} placeholder="Nome do juiz" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Valor da Causa (R$)</label>
            <input type="number" className="input-field text-sm" value={form.estimated_value} onChange={(e) => setForm({ ...form, estimated_value: e.target.value })} />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Tipo de Ação</label>
            <input className="input-field text-sm" value={form.tipo_acao} onChange={(e) => setForm({ ...form, tipo_acao: e.target.value })} placeholder="Ex: Cobrança, Indenização..." />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Nº de Partes</label>
            <input type="number" className="input-field text-sm" value={form.num_partes} onChange={(e) => setForm({ ...form, num_partes: e.target.value })} />
          </div>
        </div>
        <div className="mt-4 flex justify-end">
          <button onClick={handlePredict} disabled={loading} className="btn-primary text-sm">
            {loading ? 'Analisando...' : 'Prever Resultado'}
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-sm font-medium text-gray-700 mb-4 flex items-center gap-2">
            <Scale className="h-4 w-4" />
            Resultado da Predição
          </h3>

          {result.warning && (
            <div className="mb-4 p-3 rounded-lg bg-yellow-50 border border-yellow-200 flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5 shrink-0" />
              <p className="text-sm text-yellow-800">{result.warning}</p>
            </div>
          )}

          {result.outcome && (
            <>
              <div className="text-center mb-6">
                <p className="text-sm text-gray-500">Resultado mais provável</p>
                <p className={`text-2xl font-bold mt-1 ${outcomeColor[result.outcome] || 'text-gray-900'}`}>
                  {outcomeLabel[result.outcome] || result.outcome}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  Confiança: {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>

              <div className="space-y-3 mb-6">
                <h4 className="text-xs font-medium text-gray-500 uppercase">Probabilidades</h4>
                {Object.entries(result.probabilities).map(([key, value]) => (
                  <div key={key}>
                    <div className="flex justify-between text-sm mb-1">
                      <span>{outcomeLabel[key] || key}</span>
                      <span className="font-medium">{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-gray-100 rounded-full">
                      <div
                        className={`h-2 rounded-full ${
                          key === 'procedente' ? 'bg-green-500' :
                          key === 'parcialmente_procedente' ? 'bg-yellow-500' :
                          key === 'improcedente' ? 'bg-red-500' : 'bg-gray-400'
                        }`}
                        style={{ width: `${value * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {result.factors.length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-gray-500 uppercase mb-2">Fatores</h4>
                  <div className="space-y-1">
                    {result.factors.map((f, i) => (
                      <div key={i} className="flex justify-between text-sm border-b border-gray-50 pb-1">
                        <span className="text-gray-600">{f.name}</span>
                        <span className="font-medium text-gray-900">{String(f.value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
