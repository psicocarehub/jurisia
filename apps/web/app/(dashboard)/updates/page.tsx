'use client';

import { useEffect, useState, useCallback } from 'react';
import { apiFetch } from '@/lib/api';
import { useToast } from '@/components/toast';
import {
  Newspaper,
  Scale,
  FileText,
  BookOpen,
  ClipboardList,
  Gavel,
  Star,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  Search,
  Loader2,
  Filter,
  TrendingUp,
  Globe,
  RefreshCw,
  X,
  Calendar,
  Building2,
} from 'lucide-react';

interface ContentUpdate {
  id: string;
  source: string;
  category: string;
  subcategory?: string;
  title: string;
  summary?: string;
  content_preview?: string;
  areas: string[];
  court_or_organ?: string;
  territory?: string;
  publication_date?: string;
  source_url?: string;
  relevance_score: number;
  is_verified: boolean;
  captured_at: string;
  metadata: Record<string, unknown>;
}

interface FeedResponse {
  items: ContentUpdate[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

interface StatsResponse {
  date_from: string;
  date_to: string;
  total: number;
  by_category: Record<string, number>;
  by_source: Record<string, number>;
  by_territory: Record<string, number>;
}

interface SourceInfo {
  source: string;
  last_run: string;
  last_status: string;
  last_error?: string;
  total_records: number;
  runs: number;
}

const CATEGORY_CONFIG: Record<string, { label: string; icon: React.ReactNode; color: string }> = {
  legislacao: {
    label: 'Legislação',
    icon: <FileText className="h-5 w-5" />,
    color: 'bg-blue-50 text-blue-700 border-blue-200',
  },
  jurisprudencia: {
    label: 'Jurisprudência',
    icon: <Gavel className="h-5 w-5" />,
    color: 'bg-purple-50 text-purple-700 border-purple-200',
  },
  doutrina: {
    label: 'Doutrina',
    icon: <BookOpen className="h-5 w-5" />,
    color: 'bg-green-50 text-green-700 border-green-200',
  },
  normativo: {
    label: 'Normativos',
    icon: <ClipboardList className="h-5 w-5" />,
    color: 'bg-orange-50 text-orange-700 border-orange-200',
  },
  parecer: {
    label: 'Pareceres',
    icon: <Scale className="h-5 w-5" />,
    color: 'bg-teal-50 text-teal-700 border-teal-200',
  },
  sumula: {
    label: 'Súmulas',
    icon: <Star className="h-5 w-5" />,
    color: 'bg-yellow-50 text-yellow-700 border-yellow-200',
  },
  outro: {
    label: 'Outros',
    icon: <Newspaper className="h-5 w-5" />,
    color: 'bg-gray-50 text-gray-700 border-gray-200',
  },
};

type DateRange = 'today' | 'week' | 'month';

export default function UpdatesPage() {
  const [feed, setFeed] = useState<FeedResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [highlights, setHighlights] = useState<ContentUpdate[]>([]);
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [dateRange, setDateRange] = useState<DateRange>('today');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedArea, setSelectedArea] = useState<string>('');
  const [selectedTerritory, setSelectedTerritory] = useState<string>('');
  const [searchText, setSearchText] = useState<string>('');
  const [page, setPage] = useState(1);
  const [showSources, setShowSources] = useState(false);
  const [selectedItem, setSelectedItem] = useState<ContentUpdate | null>(null);
  const { error: showError } = useToast();

  const getDateRange = useCallback((): { from: string; to: string } => {
    const today = new Date();
    const to = today.toISOString().split('T')[0];
    let from = to;
    if (dateRange === 'week') {
      const d = new Date(today);
      d.setDate(d.getDate() - 7);
      from = d.toISOString().split('T')[0];
    } else if (dateRange === 'month') {
      const d = new Date(today);
      d.setDate(d.getDate() - 30);
      from = d.toISOString().split('T')[0];
    }
    return { from, to };
  }, [dateRange]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    const { from, to } = getDateRange();

    const params = new URLSearchParams({
      date_from: from,
      date_to: to,
      page: String(page),
      per_page: '30',
    });
    if (selectedCategory) params.set('category', selectedCategory);
    if (selectedArea) params.set('area', selectedArea);
    if (selectedTerritory) params.set('territory', selectedTerritory);
    if (searchText) params.set('search', searchText);

    try {
      const [feedRes, statsRes, highlightsRes] = await Promise.all([
        apiFetch(`/api/v1/updates/feed?${params}`),
        apiFetch(`/api/v1/updates/stats?date_from=${from}&date_to=${to}`),
        apiFetch(`/api/v1/updates/highlights?days=${dateRange === 'today' ? 1 : dateRange === 'week' ? 7 : 30}`),
      ]);

      if (feedRes.ok) setFeed(await feedRes.json());
      if (statsRes.ok) setStats(await statsRes.json());
      if (highlightsRes.ok) {
        const h = await highlightsRes.json();
        setHighlights(h.highlights || []);
      }
    } catch {
      showError('Erro ao carregar novidades');
    } finally {
      setLoading(false);
    }
  }, [getDateRange, page, selectedCategory, selectedArea, selectedTerritory, searchText]);

  const fetchSources = useCallback(async () => {
    try {
      const res = await apiFetch('/api/v1/updates/sources');
      if (res.ok) {
        const data = await res.json();
        setSources(data.sources || []);
      }
    } catch {
      showError('Erro ao carregar fontes');
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (showSources) fetchSources();
  }, [showSources, fetchSources]);

  const clearFilters = () => {
    setSelectedCategory(null);
    setSelectedArea('');
    setSelectedTerritory('');
    setSearchText('');
    setPage(1);
  };

  const hasFilters = selectedCategory || selectedArea || selectedTerritory || searchText;

  const formatDate = (d?: string | null) => {
    if (!d) return '—';
    try {
      return new Date(d).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit', year: 'numeric' });
    } catch {
      return d;
    }
  };

  const formatTime = (d?: string | null) => {
    if (!d) return '';
    try {
      return new Date(d).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  const rangeLabel: Record<DateRange, string> = {
    today: 'Hoje',
    week: 'Semana',
    month: 'Mês',
  };

  return (
    <div className="flex h-full">
      {/* Main content */}
      <div className={`flex-1 overflow-auto p-6 ${selectedItem ? 'pr-0' : ''}`}>
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Novidades Jurídicas</h1>
            <p className="mt-1 text-sm text-gray-500">
              Acompanhe tudo de novo capturado pelas nossas fontes
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
            >
              <Globe className="h-4 w-4" />
              Fontes
            </button>
            <button
              onClick={() => { setPage(1); fetchData(); }}
              className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
            >
              <RefreshCw className="h-4 w-4" />
              Atualizar
            </button>
            <div className="flex rounded-lg border border-gray-200 bg-white">
              {(['today', 'week', 'month'] as DateRange[]).map((r) => (
                <button
                  key={r}
                  onClick={() => { setDateRange(r); setPage(1); }}
                  className={`px-3 py-2 text-sm font-medium ${
                    dateRange === r
                      ? 'bg-legal-blue-50 text-legal-blue-700'
                      : 'text-gray-600 hover:bg-gray-50'
                  } ${r === 'today' ? 'rounded-l-lg' : ''} ${r === 'month' ? 'rounded-r-lg' : ''}`}
                >
                  {rangeLabel[r]}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Stats cards */}
        {stats && (
          <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-7">
            {Object.entries(CATEGORY_CONFIG).map(([key, cfg]) => {
              const count = stats.by_category[key] || 0;
              return (
                <button
                  key={key}
                  onClick={() => {
                    setSelectedCategory(selectedCategory === key ? null : key);
                    setPage(1);
                  }}
                  className={`flex flex-col items-center rounded-xl border p-3 transition-all ${
                    selectedCategory === key
                      ? `${cfg.color} ring-2 ring-offset-1`
                      : 'border-gray-200 bg-white hover:border-gray-300'
                  }`}
                >
                  <span className={selectedCategory === key ? '' : 'text-gray-400'}>
                    {cfg.icon}
                  </span>
                  <span className="mt-1 text-xl font-bold">{count}</span>
                  <span className="text-xs">{cfg.label}</span>
                </button>
              );
            })}
          </div>
        )}

        {/* Highlights */}
        {highlights.length > 0 && !hasFilters && (
          <div className="mb-6">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-700">
              <TrendingUp className="h-4 w-4 text-orange-500" />
              Destaques
            </h2>
            <div className="flex gap-3 overflow-x-auto pb-2">
              {highlights.slice(0, 5).map((h) => {
                const cfg = CATEGORY_CONFIG[h.category] || CATEGORY_CONFIG.outro;
                return (
                  <button
                    key={h.id}
                    onClick={() => setSelectedItem(h)}
                    className="flex min-w-[280px] max-w-[320px] flex-col rounded-xl border border-gray-200 bg-white p-4 text-left hover:border-gray-300 hover:shadow-sm"
                  >
                    <div className="mb-2 flex items-center gap-2">
                      <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${cfg.color}`}>
                        {cfg.label}
                      </span>
                      {h.court_or_organ && (
                        <span className="text-xs text-gray-500">{h.court_or_organ}</span>
                      )}
                    </div>
                    <p className="line-clamp-2 text-sm font-medium text-gray-900">{h.title}</p>
                    {h.summary && (
                      <p className="mt-1 line-clamp-2 text-xs text-gray-500">{h.summary}</p>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Filters bar */}
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              value={searchText}
              onChange={(e) => { setSearchText(e.target.value); setPage(1); }}
              placeholder="Buscar por título..."
              className="w-full rounded-lg border border-gray-200 bg-white py-2 pl-9 pr-3 text-sm focus:border-legal-blue-400 focus:outline-none focus:ring-1 focus:ring-legal-blue-400"
            />
          </div>
          <select
            value={selectedArea}
            onChange={(e) => { setSelectedArea(e.target.value); setPage(1); }}
            className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 focus:border-legal-blue-400 focus:outline-none"
          >
            <option value="">Todas as áreas</option>
            <option value="tributario">Tributário</option>
            <option value="aduaneiro">Aduaneiro</option>
            <option value="civil">Civil</option>
            <option value="penal">Penal</option>
            <option value="trabalhista">Trabalhista</option>
            <option value="administrativo">Administrativo</option>
            <option value="constitucional">Constitucional</option>
            <option value="ambiental">Ambiental</option>
            <option value="empresarial">Empresarial</option>
            <option value="processual">Processual</option>
            <option value="regulatorio">Regulatório</option>
            <option value="saude">Saúde</option>
            <option value="mercado_financeiro">Mercado Financeiro</option>
          </select>
          <select
            value={selectedTerritory}
            onChange={(e) => { setSelectedTerritory(e.target.value); setPage(1); }}
            className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 focus:border-legal-blue-400 focus:outline-none"
          >
            <option value="">Todos os territórios</option>
            <option value="federal">Federal</option>
            <option value="SP">São Paulo</option>
            <option value="RJ">Rio de Janeiro</option>
            <option value="MG">Minas Gerais</option>
            <option value="RS">Rio Grande do Sul</option>
            <option value="PR">Paraná</option>
            <option value="SC">Santa Catarina</option>
            <option value="BA">Bahia</option>
            <option value="DF">Distrito Federal</option>
          </select>
          {hasFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-500 hover:bg-gray-50"
            >
              <X className="h-3 w-3" />
              Limpar
            </button>
          )}
        </div>

        {/* Feed list */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-6 w-6 animate-spin text-legal-blue-600" />
            <span className="ml-2 text-sm text-gray-500">Carregando novidades...</span>
          </div>
        ) : feed && feed.items.length > 0 ? (
          <>
            <div className="space-y-2">
              {feed.items.map((item) => {
                const cfg = CATEGORY_CONFIG[item.category] || CATEGORY_CONFIG.outro;
                const isSelected = selectedItem?.id === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setSelectedItem(item)}
                    className={`w-full rounded-xl border bg-white p-4 text-left transition-all hover:shadow-sm ${
                      isSelected
                        ? 'border-legal-blue-300 ring-1 ring-legal-blue-200'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="mb-1.5 flex flex-wrap items-center gap-2">
                          <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${cfg.color}`}>
                            {cfg.label}
                          </span>
                          {item.subcategory && (
                            <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
                              {item.subcategory.replace(/_/g, ' ')}
                            </span>
                          )}
                          {item.areas.slice(0, 3).map((a) => (
                            <span key={a} className="rounded-full bg-gray-50 px-2 py-0.5 text-xs text-gray-500">
                              {a}
                            </span>
                          ))}
                        </div>
                        <p className="text-sm font-medium text-gray-900">{item.title}</p>
                        {item.summary && (
                          <p className="mt-1 line-clamp-2 text-xs text-gray-500">{item.summary}</p>
                        )}
                      </div>
                      <div className="flex shrink-0 flex-col items-end gap-1">
                        {item.court_or_organ && (
                          <span className="flex items-center gap-1 text-xs text-gray-500">
                            <Building2 className="h-3 w-3" />
                            {item.court_or_organ}
                          </span>
                        )}
                        <span className="flex items-center gap-1 text-xs text-gray-400">
                          <Calendar className="h-3 w-3" />
                          {formatDate(item.publication_date || item.captured_at)}
                        </span>
                        {item.source_url && (
                          <a
                            href={item.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="flex items-center gap-1 text-xs text-legal-blue-600 hover:underline"
                          >
                            <ExternalLink className="h-3 w-3" />
                            Verificar
                          </a>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Pagination */}
            {feed.total_pages > 1 && (
              <div className="mt-4 flex items-center justify-between">
                <span className="text-sm text-gray-500">
                  {feed.total} resultado{feed.total !== 1 ? 's' : ''} &middot; Página {feed.page} de {feed.total_pages}
                </span>
                <div className="flex gap-1">
                  <button
                    onClick={() => setPage(Math.max(1, page - 1))}
                    disabled={page <= 1}
                    className="rounded-lg border border-gray-200 p-2 text-gray-600 hover:bg-gray-50 disabled:opacity-40"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => setPage(Math.min(feed.total_pages, page + 1))}
                    disabled={page >= feed.total_pages}
                    className="rounded-lg border border-gray-200 p-2 text-gray-600 hover:bg-gray-50 disabled:opacity-40"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center py-20 text-center">
            <Newspaper className="h-12 w-12 text-gray-300" />
            <p className="mt-3 text-sm font-medium text-gray-700">Nenhuma novidade encontrada</p>
            <p className="mt-1 text-xs text-gray-500">
              {hasFilters
                ? 'Tente ajustar os filtros ou o período'
                : 'As novidades aparecerão aqui quando os pipelines de ingestão capturarem conteúdo'}
            </p>
          </div>
        )}
      </div>

      {/* Detail panel */}
      {selectedItem && (
        <div className="w-[400px] shrink-0 overflow-auto border-l border-gray-200 bg-white p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Detalhes</h3>
            <button
              onClick={() => setSelectedItem(null)}
              className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="space-y-4">
            {/* Category & subcategory */}
            <div>
              <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${(CATEGORY_CONFIG[selectedItem.category] || CATEGORY_CONFIG.outro).color}`}>
                {(CATEGORY_CONFIG[selectedItem.category] || CATEGORY_CONFIG.outro).icon}
                {(CATEGORY_CONFIG[selectedItem.category] || CATEGORY_CONFIG.outro).label}
              </span>
              {selectedItem.subcategory && (
                <span className="ml-2 rounded-full bg-gray-100 px-2.5 py-1 text-xs text-gray-600">
                  {selectedItem.subcategory.replace(/_/g, ' ')}
                </span>
              )}
            </div>

            {/* Title */}
            <h2 className="text-base font-semibold text-gray-900">{selectedItem.title}</h2>

            {/* Metadata */}
            <div className="space-y-2 text-sm">
              {selectedItem.court_or_organ && (
                <div className="flex items-center gap-2 text-gray-600">
                  <Building2 className="h-4 w-4 text-gray-400" />
                  <span>{selectedItem.court_or_organ}</span>
                </div>
              )}
              {selectedItem.territory && (
                <div className="flex items-center gap-2 text-gray-600">
                  <Globe className="h-4 w-4 text-gray-400" />
                  <span>{selectedItem.territory === 'federal' ? 'Federal' : selectedItem.territory}</span>
                </div>
              )}
              <div className="flex items-center gap-2 text-gray-600">
                <Calendar className="h-4 w-4 text-gray-400" />
                <span>
                  Publicação: {formatDate(selectedItem.publication_date)} &middot; Capturado: {formatDate(selectedItem.captured_at)} {formatTime(selectedItem.captured_at)}
                </span>
              </div>
            </div>

            {/* Areas */}
            {selectedItem.areas.length > 0 && (
              <div>
                <p className="mb-1.5 text-xs font-medium text-gray-500">Áreas do Direito</p>
                <div className="flex flex-wrap gap-1.5">
                  {selectedItem.areas.map((a) => (
                    <span key={a} className="rounded-full bg-legal-blue-50 px-2.5 py-0.5 text-xs font-medium text-legal-blue-700">
                      {a}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Summary */}
            {selectedItem.summary && (
              <div>
                <p className="mb-1.5 text-xs font-medium text-gray-500">Resumo / Ementa</p>
                <p className="text-sm leading-relaxed text-gray-700">{selectedItem.summary}</p>
              </div>
            )}

            {/* Content preview */}
            {selectedItem.content_preview && selectedItem.content_preview !== selectedItem.summary && (
              <div>
                <p className="mb-1.5 text-xs font-medium text-gray-500">Prévia do Conteúdo</p>
                <p className="text-sm leading-relaxed text-gray-600">{selectedItem.content_preview}</p>
              </div>
            )}

            {/* Source */}
            <div>
              <p className="mb-1.5 text-xs font-medium text-gray-500">Fonte</p>
              <p className="text-sm text-gray-600">{selectedItem.source.replace(/_/g, ' ')}</p>
            </div>

            {/* Verify link */}
            {selectedItem.source_url && (
              <a
                href={selectedItem.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-legal-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-legal-blue-700"
              >
                <ExternalLink className="h-4 w-4" />
                Verificar na Fonte Original
              </a>
            )}
          </div>
        </div>
      )}

      {/* Sources panel */}
      {showSources && (
        <div className="w-[350px] shrink-0 overflow-auto border-l border-gray-200 bg-white p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Status das Fontes</h3>
            <button
              onClick={() => setShowSources(false)}
              className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {sources.length > 0 ? (
            <div className="space-y-2">
              {sources.map((s) => (
                <div
                  key={s.source}
                  className="rounded-lg border border-gray-100 bg-gray-50 p-3"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-800">
                      {s.source.replace(/_/g, ' ')}
                    </span>
                    <span
                      className={`h-2 w-2 rounded-full ${
                        s.last_status === 'completed' ? 'bg-green-500' : s.last_status === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
                      }`}
                    />
                  </div>
                  <div className="mt-1 text-xs text-gray-500">
                    <span>{s.total_records.toLocaleString('pt-BR')} registros</span>
                    <span className="mx-1">&middot;</span>
                    <span>{formatDate(s.last_run)} {formatTime(s.last_run)}</span>
                  </div>
                  {s.last_error && (
                    <p className="mt-1 line-clamp-1 text-xs text-red-500">{s.last_error}</p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center py-10 text-center">
              <Globe className="h-8 w-8 text-gray-300" />
              <p className="mt-2 text-sm text-gray-500">Carregando fontes...</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
