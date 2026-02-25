'use client';

import { useEffect, useState, useCallback } from 'react';
import { apiFetch } from '@/lib/api';
import { useToast } from '@/components/toast';
import {
  Bell,
  CheckCheck,
  ChevronRight,
  X,
  AlertTriangle,
  FileText,
  Scale,
  Clock,
} from 'lucide-react';

interface Alert {
  id: string;
  type: string;
  title: string;
  description: string;
  area?: string;
  severity: string;
  is_read: boolean;
  created_at: string;
  metadata?: Record<string, unknown>;
}

const TYPE_ICONS: Record<string, React.ReactNode> = {
  law_change: <FileText className="h-5 w-5 text-blue-500" />,
  new_thesis: <Scale className="h-5 w-5 text-purple-500" />,
  deadline: <Clock className="h-5 w-5 text-orange-500" />,
  warning: <AlertTriangle className="h-5 w-5 text-yellow-500" />,
};

const SEVERITY_COLORS: Record<string, string> = {
  high: 'bg-red-50 text-red-700 border-red-200',
  medium: 'bg-yellow-50 text-yellow-700 border-yellow-200',
  low: 'bg-blue-50 text-blue-700 border-blue-200',
};

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');
  const [selected, setSelected] = useState<Alert | null>(null);
  const { error: showError } = useToast();

  const fetchAlerts = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiFetch('/api/v1/alerts');
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.alerts || []);
      }
    } catch {
      showError('Erro ao carregar alertas');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  const markRead = async (id: string) => {
    try {
      await apiFetch(`/api/v1/alerts/${id}/read`, { method: 'PATCH' });
      setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, is_read: true } : a)));
      if (selected?.id === id) setSelected({ ...selected, is_read: true });
    } catch {
      showError('Erro ao marcar alerta como lido');
    }
  };

  const markAllRead = async () => {
    try {
      await apiFetch('/api/v1/alerts/read-all', { method: 'PATCH' });
      setAlerts((prev) => prev.map((a) => ({ ...a, is_read: true })));
    } catch {
      showError('Erro ao marcar alertas como lidos');
    }
  };

  const filtered = filter === 'unread' ? alerts.filter((a) => !a.is_read) : alerts;
  const unreadCount = alerts.filter((a) => !a.is_read).length;

  return (
    <div className="flex h-full">
      <div className={`${selected ? 'w-1/2 border-r' : 'w-full'} flex flex-col bg-white`}>
        <div className="border-b p-4">
          <div className="flex items-center justify-between mb-3">
            <h1 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <Bell className="h-5 w-5 text-legal-blue-600" />
              Alertas
              {unreadCount > 0 && (
                <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded-full font-medium">
                  {unreadCount}
                </span>
              )}
            </h1>
            {unreadCount > 0 && (
              <button onClick={markAllRead} className="text-sm text-legal-blue-600 hover:underline flex items-center gap-1">
                <CheckCheck className="h-4 w-4" />
                Marcar todos como lidos
              </button>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-3 py-1.5 text-sm rounded-lg ${filter === 'all' ? 'bg-legal-blue-600 text-white' : 'bg-gray-100 text-gray-600'}`}
            >
              Todos ({alerts.length})
            </button>
            <button
              onClick={() => setFilter('unread')}
              className={`px-3 py-1.5 text-sm rounded-lg ${filter === 'unread' ? 'bg-legal-blue-600 text-white' : 'bg-gray-100 text-gray-600'}`}
            >
              Não lidos ({unreadCount})
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="p-8 text-center text-gray-400">Carregando...</div>
          ) : filtered.length === 0 ? (
            <div className="p-12 text-center">
              <Bell className="h-12 w-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">Nenhum alerta</p>
            </div>
          ) : (
            <div className="divide-y">
              {filtered.map((alert) => (
                <div
                  key={alert.id}
                  onClick={() => { setSelected(alert); if (!alert.is_read) markRead(alert.id); }}
                  className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                    !alert.is_read ? 'bg-blue-50/50' : ''
                  } ${selected?.id === alert.id ? 'bg-legal-blue-50' : ''}`}
                >
                  <div className="flex items-start gap-3">
                    {TYPE_ICONS[alert.type] || TYPE_ICONS.warning}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className={`text-sm truncate ${!alert.is_read ? 'font-semibold' : 'font-medium'} text-gray-900`}>
                          {alert.title}
                        </h3>
                        {!alert.is_read && <span className="h-2 w-2 bg-blue-500 rounded-full shrink-0" />}
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5 line-clamp-1">{alert.description}</p>
                      <div className="flex gap-2 mt-1.5">
                        {alert.severity && (
                          <span className={`text-xs px-1.5 py-0.5 rounded border ${SEVERITY_COLORS[alert.severity] || SEVERITY_COLORS.low}`}>
                            {alert.severity === 'high' ? 'Alta' : alert.severity === 'medium' ? 'Média' : 'Baixa'}
                          </span>
                        )}
                        {alert.area && <span className="text-xs text-gray-400">{alert.area}</span>}
                        <span className="text-xs text-gray-400">{new Date(alert.created_at).toLocaleDateString('pt-BR')}</span>
                      </div>
                    </div>
                    <ChevronRight className="h-4 w-4 text-gray-300 shrink-0 mt-1" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {selected && (
        <div className="w-1/2 flex flex-col bg-white">
          <div className="border-b p-4 flex justify-between items-center">
            <h2 className="font-semibold text-gray-900 text-sm truncate">{selected.title}</h2>
            <button onClick={() => setSelected(null)} className="text-gray-400 hover:text-gray-600">
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex-1 overflow-auto p-6">
            <div className="flex items-center gap-3 mb-4">
              {TYPE_ICONS[selected.type] || TYPE_ICONS.warning}
              <div>
                <p className="text-xs text-gray-500">
                  {selected.type === 'law_change' ? 'Mudança Legislativa' :
                   selected.type === 'new_thesis' ? 'Nova Tese' :
                   selected.type === 'deadline' ? 'Prazo' : 'Aviso'}
                </p>
                <p className="text-xs text-gray-400">{new Date(selected.created_at).toLocaleString('pt-BR')}</p>
              </div>
            </div>
            <div className="prose prose-sm max-w-none">
              <p className="text-gray-700 whitespace-pre-wrap">{selected.description}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
