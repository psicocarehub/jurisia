'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import { apiPost } from '@/lib/api';
import { cn } from '@/lib/utils';

export interface SearchFilters {
  area?: string;
  tribunal?: string;
  dateFrom?: string;
  dateTo?: string;
}

export interface SearchResult {
  id: string;
  title: string;
  court?: string;
  date?: string;
  doc_type?: string;
  content?: string;
  score?: number;
  highlight?: string;
}

interface SearchBarProps {
  onResults?: (results: SearchResult[]) => void;
  onSearching?: (searching: boolean) => void;
  placeholder?: string;
  className?: string;
}

const DEBOUNCE_MS = 300;

export function SearchBar({
  onResults,
  onSearching,
  placeholder = 'Buscar legislação e jurisprudência...',
  className,
}: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [area, setArea] = useState('');
  const [tribunal, setTribunal] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  const performSearch = useCallback(
    async (searchQuery: string) => {
      if (!searchQuery.trim()) {
        onResults?.([]);
        return;
      }

      setIsSearching(true);
      onSearching?.(true);

      try {
        const results = await apiPost<{ results?: SearchResult[] }>(
          '/api/v1/search',
          {
            query: searchQuery.trim(),
            area: area || undefined,
            tribunal: tribunal || undefined,
            date_from: dateFrom || undefined,
            date_to: dateTo || undefined,
          }
        );
        onResults?.(results.results ?? []);
      } catch {
        onResults?.([]);
      } finally {
        setIsSearching(false);
        onSearching?.(false);
      }
    },
    [area, tribunal, dateFrom, dateTo, onResults, onSearching]
  );

  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  const handleQueryChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setQuery(value);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => performSearch(value), DEBOUNCE_MS);
    },
    [performSearch]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = undefined;
      }
      performSearch(query);
    },
    [query, performSearch]
  );

  return (
    <form onSubmit={handleSubmit} className={cn('space-y-3', className)}>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
        <input
          type="search"
          value={query}
          onChange={handleQueryChange}
          placeholder={placeholder}
          aria-label="Buscar"
          className="input-field pl-10 pr-4"
        />
        {isSearching && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500">
            Buscando...
          </span>
        )}
      </div>

      <div className="flex flex-wrap gap-3">
        <label className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Área:</span>
          <select
            value={area}
            onChange={(e) => setArea(e.target.value)}
            className="input-field w-40 py-2 text-sm"
          >
            <option value="">Todas</option>
            <option value="civil">Civil</option>
            <option value="penal">Penal</option>
            <option value="trabalhista">Trabalhista</option>
            <option value="administrativo">Administrativo</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Tribunal:</span>
          <select
            value={tribunal}
            onChange={(e) => setTribunal(e.target.value)}
            className="input-field w-48 py-2 text-sm"
          >
            <option value="">Todos</option>
            <option value="stf">STF</option>
            <option value="stj">STJ</option>
            <option value="tst">TST</option>
            <option value="trf">TRF</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-sm text-gray-500">De:</span>
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="input-field w-36 py-2 text-sm"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Até:</span>
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="input-field w-36 py-2 text-sm"
          />
        </label>
      </div>
    </form>
  );
}
