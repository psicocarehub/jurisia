'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { apiFetch, apiPost } from '@/lib/api';
import { useToast } from '@/components/toast';
import {
  Briefcase,
  Plus,
  Search,
  ChevronRight,
  X,
  Trash2,
  Edit3,
  Filter,
  Scan,
} from 'lucide-react';

interface Case {
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

const AREAS = [
  'Cível', 'Trabalhista', 'Criminal', 'Tributário', 'Família',
  'Consumidor', 'Administrativo', 'Previdenciário', 'Empresarial',
];

const STATUSES = ['active', 'archived', 'closed', 'pending'];

export default function CasesPage() {
  const router = useRouter();
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [selectedCase, setSelectedCase] = useState<Case | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterArea, setFilterArea] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const { error: showError, success: showSuccess } = useToast();

  const fetchCases = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filterArea) params.set('area', filterArea);
      if (filterStatus) params.set('status', filterStatus);
      const res = await apiFetch(`/api/v1/cases?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setCases(data.cases || []);
      }
    } catch (e) {
      showError('Erro ao carregar casos');
    } finally {
      setLoading(false);
    }
  }, [filterArea, filterStatus]);

  useEffect(() => {
    fetchCases();
  }, [fetchCases]);

  const filteredCases = cases.filter((c) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      c.title.toLowerCase().includes(q) ||
      (c.cnj_number || '').toLowerCase().includes(q) ||
      (c.client_name || '').toLowerCase().includes(q)
    );
  });

  const handleDelete = async (id: string) => {
    if (!confirm('Tem certeza que deseja excluir este caso?')) return;
    try {
      await apiFetch(`/api/v1/cases/${id}`, { method: 'DELETE' });
      setCases((prev) => prev.filter((c) => c.id !== id));
      if (selectedCase?.id === id) setSelectedCase(null);
      showSuccess('Caso excluído');
    } catch {
      showError('Erro ao excluir caso');
    }
  };

  return (
    <div className="flex h-full">
      {/* List panel */}
      <div className={`${selectedCase ? 'w-1/2 border-r' : 'w-full'} flex flex-col bg-white`}>
        <div className="border-b p-4">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <Briefcase className="h-5 w-5 text-legal-blue-600" />
              Casos
            </h1>
            <button
              onClick={() => setShowCreate(true)}
              className="btn-primary flex items-center gap-1.5 text-sm"
            >
              <Plus className="h-4 w-4" />
              Novo Caso
            </button>
          </div>

          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Buscar por título, CNJ ou cliente..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input-field pl-9 text-sm"
              />
            </div>
            <select
              value={filterArea}
              onChange={(e) => setFilterArea(e.target.value)}
              className="input-field text-sm w-36"
            >
              <option value="">Todas áreas</option>
              {AREAS.map((a) => (
                <option key={a} value={a.toLowerCase()}>{a}</option>
              ))}
            </select>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="input-field text-sm w-32"
            >
              <option value="">Todos status</option>
              {STATUSES.map((s) => (
                <option key={s} value={s}>{s === 'active' ? 'Ativo' : s === 'archived' ? 'Arquivado' : s === 'closed' ? 'Encerrado' : 'Pendente'}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="p-8 text-center text-gray-400">Carregando...</div>
          ) : filteredCases.length === 0 ? (
            <div className="p-8 text-center text-gray-400">
              {cases.length === 0
                ? 'Nenhum caso cadastrado. Crie o primeiro!'
                : 'Nenhum caso encontrado com os filtros atuais.'}
            </div>
          ) : (
            <div className="divide-y">
              {filteredCases.map((c) => (
                <div
                  key={c.id}
                  onClick={() => setSelectedCase(c)}
                  className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                    selectedCase?.id === c.id ? 'bg-legal-blue-50 border-l-2 border-legal-blue-600' : ''
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-gray-900 truncate">{c.title}</h3>
                      <div className="flex gap-3 mt-1 text-xs text-gray-500">
                        {c.cnj_number && <span>{c.cnj_number}</span>}
                        {c.area && (
                          <span className="px-1.5 py-0.5 bg-legal-blue-50 text-legal-blue-700 rounded">
                            {c.area}
                          </span>
                        )}
                        <span className={`px-1.5 py-0.5 rounded ${
                          c.status === 'active' ? 'bg-green-50 text-green-700' :
                          c.status === 'closed' ? 'bg-red-50 text-red-700' :
                          'bg-gray-100 text-gray-600'
                        }`}>
                          {c.status === 'active' ? 'Ativo' : c.status === 'closed' ? 'Encerrado' : c.status}
                        </span>
                      </div>
                      {c.client_name && (
                        <p className="text-xs text-gray-400 mt-1">Cliente: {c.client_name}</p>
                      )}
                    </div>
                    <ChevronRight className="h-4 w-4 text-gray-300 shrink-0 mt-1" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Detail panel */}
      {selectedCase && (
        <div className="w-1/2 flex flex-col bg-white">
          <div className="border-b p-4 flex justify-between items-center">
            <h2 className="font-semibold text-gray-900 truncate">{selectedCase.title}</h2>
            <div className="flex gap-1">
              <button
                onClick={() => router.push(`/cases/${selectedCase.id}`)}
                className="px-3 py-1.5 text-xs font-medium text-white bg-legal-blue-600 hover:bg-legal-blue-700 rounded-lg flex items-center gap-1.5 transition-colors"
              >
                <Scan className="h-3.5 w-3.5" />
                Raio-X
              </button>
              <button
                onClick={() => handleDelete(selectedCase.id)}
                className="p-2 text-gray-400 hover:text-red-500 rounded"
              >
                <Trash2 className="h-4 w-4" />
              </button>
              <button
                onClick={() => setSelectedCase(null)}
                className="p-2 text-gray-400 hover:text-gray-600 rounded"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-auto p-6 space-y-6">
            <div className="grid grid-cols-2 gap-4">
              {[
                ['Número CNJ', selectedCase.cnj_number],
                ['Área', selectedCase.area],
                ['Status', selectedCase.status],
                ['Tribunal', selectedCase.court],
                ['Juiz', selectedCase.judge_name],
                ['Cliente', selectedCase.client_name],
                ['Parte Contrária', selectedCase.opposing_party],
                ['Criado em', selectedCase.created_at?.split('T')[0]],
              ].map(([label, value]) =>
                value ? (
                  <div key={label as string}>
                    <dt className="text-xs font-medium text-gray-500 uppercase">{label}</dt>
                    <dd className="mt-1 text-sm text-gray-900">{value}</dd>
                  </div>
                ) : null
              )}
            </div>
            {selectedCase.description && (
              <div>
                <h3 className="text-xs font-medium text-gray-500 uppercase mb-1">Descrição</h3>
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{selectedCase.description}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create modal */}
      {showCreate && (
        <CreateCaseModal
          onClose={() => setShowCreate(false)}
          onCreated={(newCase) => {
            setCases((prev) => [newCase, ...prev]);
            setShowCreate(false);
          }}
        />
      )}
    </div>
  );
}

function CreateCaseModal({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (c: Case) => void;
}) {
  const { error: showError } = useToast();
  const [form, setForm] = useState({
    title: '',
    cnj_number: '',
    area: '',
    description: '',
    client_name: '',
    client_document: '',
    opposing_party: '',
    court: '',
    judge_name: '',
    estimated_value: '',
  });
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validate = (): boolean => {
    const errs: Record<string, string> = {};
    if (!form.title.trim()) errs.title = 'Título é obrigatório';
    if (form.cnj_number && !/^\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}$/.test(form.cnj_number)) {
      errs.cnj_number = 'Formato inválido (0000000-00.0000.0.00.0000)';
    }
    if (form.estimated_value && (isNaN(parseFloat(form.estimated_value)) || parseFloat(form.estimated_value) < 0)) {
      errs.estimated_value = 'Valor inválido';
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    setSaving(true);
    try {
      const body = {
        ...form,
        estimated_value: form.estimated_value ? parseFloat(form.estimated_value) : undefined,
      };
      const res = await apiPost<Case>('/api/v1/cases', body);
      onCreated(res);
    } catch {
      showError('Erro ao criar caso');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg max-h-[90vh] overflow-auto">
        <div className="flex justify-between items-center p-4 border-b">
          <h2 className="text-lg font-semibold">Novo Caso</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-4 space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Título *</label>
            <input
              className={`input-field text-sm ${errors.title ? 'border-red-400 ring-1 ring-red-400' : ''}`}
              value={form.title}
              onChange={(e) => { setForm({ ...form, title: e.target.value }); setErrors((prev) => ({ ...prev, title: '' })); }}
              required
            />
            {errors.title && <p className="text-xs text-red-500 mt-1">{errors.title}</p>}
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Número CNJ</label>
              <input
                className={`input-field text-sm ${errors.cnj_number ? 'border-red-400 ring-1 ring-red-400' : ''}`}
                placeholder="0000000-00.0000.0.00.0000"
                value={form.cnj_number}
                onChange={(e) => { setForm({ ...form, cnj_number: e.target.value }); setErrors((prev) => ({ ...prev, cnj_number: '' })); }}
              />
              {errors.cnj_number && <p className="text-xs text-red-500 mt-1">{errors.cnj_number}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Área</label>
              <select
                className="input-field text-sm"
                value={form.area}
                onChange={(e) => setForm({ ...form, area: e.target.value })}
              >
                <option value="">Selecione</option>
                {AREAS.map((a) => (
                  <option key={a} value={a.toLowerCase()}>{a}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Cliente</label>
              <input className="input-field text-sm" value={form.client_name} onChange={(e) => setForm({ ...form, client_name: e.target.value })} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Parte Contrária</label>
              <input className="input-field text-sm" value={form.opposing_party} onChange={(e) => setForm({ ...form, opposing_party: e.target.value })} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Tribunal/Vara</label>
              <input className="input-field text-sm" value={form.court} onChange={(e) => setForm({ ...form, court: e.target.value })} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Juiz</label>
              <input className="input-field text-sm" value={form.judge_name} onChange={(e) => setForm({ ...form, judge_name: e.target.value })} />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Valor da Causa (R$)</label>
            <input type="number" step="0.01" className="input-field text-sm" value={form.estimated_value} onChange={(e) => setForm({ ...form, estimated_value: e.target.value })} />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Descrição</label>
            <textarea rows={3} className="input-field text-sm" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <button type="button" onClick={onClose} className="btn-secondary text-sm">Cancelar</button>
            <button type="submit" disabled={saving} className="btn-primary text-sm">
              {saving ? 'Criando...' : 'Criar Caso'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
