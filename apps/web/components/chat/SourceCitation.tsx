'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import { BookOpen, ChevronDown, ChevronUp, Building2, Calendar, FileText } from 'lucide-react';

export interface Source {
  id?: string;
  title?: string;
  document_title?: string;
  doc_type?: string;
  court?: string;
  date?: string;
  score?: number;
  content?: string;
  snippet?: string;
}

interface SourceCitationProps {
  sources: Source[];
  className?: string;
}

const DOC_TYPE_LABELS: Record<string, { label: string; color: string }> = {
  jurisprudencia: { label: 'Jurisprudência', color: 'bg-blue-100 text-blue-800' },
  legislacao: { label: 'Legislação', color: 'bg-green-100 text-green-800' },
  sumula: { label: 'Súmula', color: 'bg-purple-100 text-purple-800' },
  doutrina: { label: 'Doutrina', color: 'bg-amber-100 text-amber-800' },
  peticao: { label: 'Petição', color: 'bg-rose-100 text-rose-800' },
};

function getDocBadge(docType?: string) {
  if (!docType) return null;
  const normalized = docType.toLowerCase().replace(/[^a-z]/g, '');
  for (const [key, val] of Object.entries(DOC_TYPE_LABELS)) {
    if (normalized.includes(key)) return val;
  }
  return { label: docType, color: 'bg-gray-100 text-gray-700' };
}

function ScoreBar({ score }: { score?: number }) {
  if (score == null) return null;
  const pct = Math.min(Math.max(score * 100, 0), 100);
  return (
    <div className="flex items-center gap-1.5" title={`Relevância: ${pct.toFixed(0)}%`}>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-gray-200">
        <div
          className={cn(
            'h-full rounded-full transition-all',
            pct > 70 ? 'bg-green-500' : pct > 40 ? 'bg-amber-500' : 'bg-red-400'
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] tabular-nums text-gray-400">{pct.toFixed(0)}%</span>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const badge = getDocBadge(source.doc_type);
  const snippet = source.snippet || source.content;
  const title = source.title || source.document_title || `Fonte ${index + 1}`;

  return (
    <div className="group rounded-xl border border-gray-200 bg-white transition-all hover:border-legal-blue-200 hover:shadow-sm">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-start gap-3 px-3.5 py-3 text-left"
      >
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-legal-blue-50 text-xs font-semibold text-legal-blue-700">
          {index + 1}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-gray-900 line-clamp-1">{title}</span>
            {badge && (
              <span className={cn('rounded-full px-2 py-0.5 text-[10px] font-medium', badge.color)}>
                {badge.label}
              </span>
            )}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-gray-500">
            {source.court && (
              <span className="flex items-center gap-1">
                <Building2 className="h-3 w-3" />
                {source.court}
              </span>
            )}
            {source.date && (
              <span className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                {source.date}
              </span>
            )}
            <ScoreBar score={source.score} />
          </div>
        </div>
        <span className="mt-0.5 text-gray-400 group-hover:text-gray-600">
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </span>
      </button>
      {expanded && snippet && (
        <div className="border-t border-gray-100 px-3.5 py-3">
          <p className="text-xs leading-relaxed text-gray-600">{snippet}</p>
        </div>
      )}
    </div>
  );
}

export function SourceCitation({ sources, className }: SourceCitationProps) {
  const [showAll, setShowAll] = useState(false);

  if (!sources?.length) return null;

  const visible = showAll ? sources : sources.slice(0, 3);
  const hasMore = sources.length > 3;

  return (
    <div className={cn('ml-11 mt-2', className)}>
      <div className="flex items-center gap-2 mb-2">
        <BookOpen className="h-3.5 w-3.5 text-legal-blue-500" />
        <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
          {sources.length} fonte{sources.length !== 1 ? 's' : ''} consultada{sources.length !== 1 ? 's' : ''}
        </span>
      </div>
      <div className="space-y-2">
        {visible.map((source, i) => (
          <SourceCard key={source.id || i} source={source} index={i} />
        ))}
      </div>
      {hasMore && (
        <button
          type="button"
          onClick={() => setShowAll(!showAll)}
          className="mt-2 flex items-center gap-1 text-xs font-medium text-legal-blue-600 hover:text-legal-blue-700"
        >
          <FileText className="h-3 w-3" />
          {showAll ? 'Mostrar menos' : `Ver todas as ${sources.length} fontes`}
        </button>
      )}
    </div>
  );
}
