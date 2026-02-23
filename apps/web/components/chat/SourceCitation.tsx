'use client';

import { cn } from '@/lib/utils';

export interface Source {
  id?: string;
  title?: string;
  document_title?: string;
  doc_type?: string;
  court?: string;
  date?: string;
  score?: number;
  content?: string;
}

interface SourceCitationProps {
  sources: Source[];
  className?: string;
}

export function SourceCitation({ sources, className }: SourceCitationProps) {
  if (!sources?.length) return null;

  return (
    <div className={cn('space-y-2', className)}>
      <p className="text-xs font-medium uppercase tracking-wide text-gray-500">
        Fontes consultadas
      </p>
      <ul className="space-y-2">
        {sources.map((source, i) => (
          <li
            key={source.id || i}
            className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm"
          >
            <span className="font-medium text-gray-700">
              {source.title || source.document_title || `Fonte ${i + 1}`}
            </span>
            {(source.court || source.doc_type) && (
              <span className="ml-2 text-gray-500">
                {[source.doc_type, source.court].filter(Boolean).join(' • ')}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
