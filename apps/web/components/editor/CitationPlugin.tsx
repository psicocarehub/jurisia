'use client';

import { FileCheck, AlertCircle, Clock, HelpCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

export type CitationType = 'legislacao' | 'jurisprudencia';

export type CitationStatus =
  | 'verified'
  | 'not_found'
  | 'revoked'
  | 'outdated'
  | 'unchecked';

export interface Citation {
  id: string;
  text: string;
  type: CitationType;
  status: CitationStatus;
}

interface CitationPluginProps {
  citations: Citation[];
  onVerifyAll: () => void;
  isVerifying?: boolean;
  onCitationClick?: (citation: Citation) => void;
  className?: string;
}

const STATUS_CONFIG: Record<
  CitationStatus,
  { label: string; color: string; icon: React.ElementType }
> = {
  verified: {
    label: 'Verificada',
    color: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    icon: FileCheck,
  },
  not_found: {
    label: 'Não encontrada',
    color: 'bg-red-100 text-red-800 border-red-200',
    icon: AlertCircle,
  },
  revoked: {
    label: 'Revogada',
    color: 'bg-amber-100 text-amber-800 border-amber-200',
    icon: AlertCircle,
  },
  outdated: {
    label: 'Desatualizada',
    color: 'bg-amber-100 text-amber-800 border-amber-200',
    icon: Clock,
  },
  unchecked: {
    label: 'Não verificada',
    color: 'bg-gray-100 text-gray-600 border-gray-200',
    icon: HelpCircle,
  },
};

const TYPE_LABELS: Record<CitationType, string> = {
  legislacao: 'Legislação',
  jurisprudencia: 'Jurisprudência',
};

export function CitationPlugin({
  citations,
  onVerifyAll,
  isVerifying = false,
  onCitationClick,
  className,
}: CitationPluginProps) {
  return (
    <div
      className={cn(
        'flex h-full flex-col border-l border-gray-200 bg-white',
        className
      )}
    >
      <div className="border-b border-gray-200 px-4 py-3">
        <h3 className="text-sm font-semibold text-gray-900">
          Citações no documento
        </h3>
        <p className="mt-0.5 text-xs text-gray-500">
          {citations.length} {citations.length === 1 ? 'citação' : 'citações'}{' '}
          encontrada{citations.length !== 1 ? 's' : ''}
        </p>
        <button
          type="button"
          onClick={onVerifyAll}
          disabled={isVerifying || citations.length === 0}
          className="mt-3 w-full rounded-lg bg-legal-blue-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-legal-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isVerifying ? 'Verificando...' : 'Verificar todas'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {citations.length === 0 ? (
          <p className="text-center text-sm text-gray-500">
            Nenhuma citação encontrada no documento.
          </p>
        ) : (
          <ul className="space-y-2">
            {citations.map((citation) => (
              <CitationItem
                key={citation.id}
                citation={citation}
                onClick={() => onCitationClick?.(citation)}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

interface CitationItemProps {
  citation: Citation;
  onClick?: () => void;
}

function CitationItem({ citation, onClick }: CitationItemProps) {
  const config = STATUS_CONFIG[citation.status];
  const Icon = config.icon;

  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className="w-full rounded-lg border border-gray-200 bg-white p-3 text-left transition-colors hover:border-gray-300 hover:bg-gray-50"
      >
        <p className="line-clamp-2 text-sm text-gray-900">{citation.text}</p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span
            className={cn(
              'inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-xs font-medium',
              config.color
            )}
          >
            <Icon className="h-3 w-3" />
            {config.label}
          </span>
          <span className="text-xs text-gray-500">
            {TYPE_LABELS[citation.type]}
          </span>
        </div>
      </button>
    </li>
  );
}
