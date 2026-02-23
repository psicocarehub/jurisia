'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  FilePlus,
  Search,
  Trash2,
  Clock,
  FileText,
  CheckCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { apiFetch } from '@/lib/api';

interface Petition {
  id: string;
  title: string;
  petition_type: string | null;
  status: string;
  ai_generated: boolean;
  created_at: string | null;
  updated_at: string | null;
  content: string | null;
}

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: typeof CheckCircle }> = {
  draft: { label: 'Rascunho', color: 'bg-gray-100 text-gray-700', icon: Clock },
  review: { label: 'Em Revisão', color: 'bg-yellow-100 text-yellow-800', icon: AlertCircle },
  final: { label: 'Finalizada', color: 'bg-green-100 text-green-800', icon: CheckCircle },
  filed: { label: 'Protocolada', color: 'bg-blue-100 text-blue-800', icon: FileText },
};

const TYPE_LABELS: Record<string, string> = {
  peticao_inicial: 'Petição Inicial',
  contestacao: 'Contestação',
  recurso: 'Recurso',
  agravo: 'Agravo',
  mandado_seguranca: 'Mandado de Segurança',
  habeas_corpus: 'Habeas Corpus',
  inicial: 'Petição Inicial',
};

export default function PetitionsPage() {
  const router = useRouter();
  const [petitions, setPetitions] = useState<Petition[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [deleting, setDeleting] = useState<string | null>(null);

  const fetchPetitions = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (statusFilter) params.set('status', statusFilter);
      const url = `/api/v1/petitions${params.toString() ? `?${params}` : ''}`;
      const resp = await apiFetch(url);
      if (resp.ok) {
        const data = await resp.json();
        setPetitions(data.petitions || []);
      }
    } catch (err) {
      console.error('Failed to fetch petitions:', err);
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    fetchPetitions();
  }, [fetchPetitions]);

  const handleDelete = async (id: string) => {
    if (!confirm('Tem certeza que deseja excluir esta petição?')) return;
    setDeleting(id);
    try {
      const resp = await apiFetch(`/api/v1/petitions/${id}`, { method: 'DELETE' });
      if (resp.ok || resp.status === 204) {
        setPetitions((prev) => prev.filter((p) => p.id !== id));
      }
    } catch (err) {
      console.error('Delete failed:', err);
    } finally {
      setDeleting(null);
    }
  };

  const filtered = petitions.filter((p) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      p.title.toLowerCase().includes(q) ||
      (p.petition_type || '').toLowerCase().includes(q)
    );
  });

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleDateString('pt-BR', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const contentPreview = (content: string | null) => {
    if (!content) return 'Sem conteúdo';
    const text = content.replace(/<[^>]*>/g, '');
    return text.length > 120 ? text.slice(0, 120) + '...' : text;
  };

  return (
    <div className="p-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Petições</h1>
          <p className="mt-1 text-sm text-gray-600">
            Crie e edite petições com assistência de IA.
          </p>
        </div>
        <Link
          href="/petitions/novo/editor"
          className="btn-primary inline-flex items-center gap-2"
        >
          <FilePlus className="h-4 w-4" />
          Nova Petição
        </Link>
      </div>

      {/* Filters */}
      <div className="mt-6 flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Buscar por título ou tipo..."
            className="input-field w-full pl-10"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="input-field w-40"
        >
          <option value="">Todos os status</option>
          <option value="draft">Rascunho</option>
          <option value="review">Em Revisão</option>
          <option value="final">Finalizada</option>
          <option value="filed">Protocolada</option>
        </select>
      </div>

      {/* Content */}
      {loading ? (
        <div className="mt-16 flex flex-col items-center justify-center text-gray-500">
          <Loader2 className="h-8 w-8 animate-spin" />
          <p className="mt-2 text-sm">Carregando petições...</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="mt-16 flex flex-col items-center justify-center text-gray-500">
          <FileText className="h-12 w-12 text-gray-300" />
          <p className="mt-4 text-lg font-medium text-gray-600">
            {searchQuery || statusFilter ? 'Nenhuma petição encontrada' : 'Nenhuma petição ainda'}
          </p>
          <p className="mt-1 text-sm text-gray-500">
            {searchQuery || statusFilter
              ? 'Tente ajustar os filtros.'
              : 'Crie sua primeira petição com assistência de IA.'}
          </p>
          {!searchQuery && !statusFilter && (
            <Link
              href="/petitions/novo/editor"
              className="btn-primary mt-4 inline-flex items-center gap-2"
            >
              <FilePlus className="h-4 w-4" />
              Criar Petição
            </Link>
          )}
        </div>
      ) : (
        <div className="mt-6 grid gap-4">
          {filtered.map((petition) => {
            const statusCfg = STATUS_CONFIG[petition.status] || STATUS_CONFIG.draft;
            const StatusIcon = statusCfg.icon;
            return (
              <div
                key={petition.id}
                className="group relative rounded-lg border border-gray-200 bg-white p-5 transition-shadow hover:shadow-md"
              >
                <div className="flex items-start justify-between">
                  <div
                    className="flex-1 cursor-pointer"
                    onClick={() => router.push(`/petitions/${petition.id}/editor`)}
                  >
                    <div className="flex items-center gap-3">
                      <h3 className="text-base font-semibold text-gray-900 group-hover:text-blue-700">
                        {petition.title || 'Sem título'}
                      </h3>
                      <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${statusCfg.color}`}>
                        <StatusIcon className="h-3 w-3" />
                        {statusCfg.label}
                      </span>
                      {petition.ai_generated && (
                        <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
                          IA
                        </span>
                      )}
                    </div>
                    <div className="mt-1 flex items-center gap-4 text-xs text-gray-500">
                      {petition.petition_type && (
                        <span>{TYPE_LABELS[petition.petition_type] || petition.petition_type}</span>
                      )}
                      <span>Criada: {formatDate(petition.created_at)}</span>
                      {petition.updated_at !== petition.created_at && (
                        <span>Editada: {formatDate(petition.updated_at)}</span>
                      )}
                    </div>
                    <p className="mt-2 text-sm text-gray-600 line-clamp-2">
                      {contentPreview(petition.content)}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(petition.id);
                    }}
                    disabled={deleting === petition.id}
                    className="ml-4 shrink-0 rounded p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-600"
                    title="Excluir"
                  >
                    {deleting === petition.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
