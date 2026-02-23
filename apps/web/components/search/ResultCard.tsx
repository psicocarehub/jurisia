'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import type { SearchResult } from './SearchBar';
import { cn } from '@/lib/utils';

interface ResultCardProps {
  result: SearchResult;
  query?: string;
  className?: string;
}

export function ResultCard({ result, query, className }: ResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const highlightPreview = result.highlight ?? result.content?.slice(0, 200) ?? '';
  const fullContent = result.content ?? '';

  const renderHighlight = (text: string) => {
    if (!query?.trim()) return text;
    const parts = text.split(new RegExp(`(${escapeRegex(query)})`, 'gi'));
    return parts.map((part, i) =>
      part.toLowerCase() === query.toLowerCase() ? (
        <mark key={i} className="bg-amber-200 font-medium">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  return (
    <article
      className={cn(
        'rounded-lg border border-gray-200 bg-white shadow-sm transition-shadow hover:shadow-md',
        className
      )}
    >
      <button
        type="button"
        onClick={() => setIsExpanded((v) => !v)}
        className="flex w-full flex-col items-start gap-2 p-4 text-left"
      >
        <div className="flex w-full items-start justify-between gap-4">
          <h3 className="flex-1 text-sm font-semibold text-gray-900">
            {result.title}
          </h3>
          {result.score != null && (
            <span className="shrink-0 rounded-full bg-legal-blue-100 px-2 py-0.5 text-xs font-medium text-legal-blue-700">
              {Math.round(result.score * 100)}%
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500">
          {result.court && (
            <span>Tribunal: {result.court}</span>
          )}
          {result.date && (
            <span>{new Date(result.date).toLocaleDateString('pt-BR')}</span>
          )}
          {result.doc_type && (
            <span className="capitalize">{result.doc_type}</span>
          )}
        </div>

        <p className="line-clamp-2 w-full text-sm text-gray-600">
          {renderHighlight(isExpanded ? fullContent : highlightPreview)}
          {!isExpanded && fullContent.length > 200 && '...'}
        </p>

        <span className="flex items-center gap-1 text-xs font-medium text-legal-blue-600">
          {isExpanded ? (
            <>
              <ChevronUp className="h-4 w-4" />
              Recolher
            </>
          ) : (
            <>
              <ChevronDown className="h-4 w-4" />
              Ver conteúdo completo
            </>
          )}
        </span>
      </button>
    </article>
  );
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
