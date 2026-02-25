'use client';

import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import { Scale, Search, BookOpen, Brain } from 'lucide-react';

const PHASES = [
  { icon: Search, text: 'Buscando na base jurídica...' },
  { icon: BookOpen, text: 'Analisando fontes relevantes...' },
  { icon: Brain, text: 'Elaborando resposta fundamentada...' },
];

interface ThinkingIndicatorProps {
  className?: string;
}

export function ThinkingIndicator({ className }: ThinkingIndicatorProps) {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setPhase((p) => (p + 1) % PHASES.length);
    }, 2500);
    return () => clearInterval(timer);
  }, []);

  const current = PHASES[phase];
  const Icon = current.icon;

  return (
    <div className={cn('flex items-start gap-3', className)}>
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-legal-blue-500 to-legal-blue-700 shadow-sm">
        <Scale className="h-4 w-4 text-white" />
      </div>
      <div className="rounded-2xl rounded-tl-sm bg-white px-4 py-3 shadow-sm ring-1 ring-gray-100">
        <div className="flex items-center gap-2.5">
          <Icon className="h-4 w-4 animate-pulse text-legal-blue-500" />
          <span className="text-sm text-gray-600">{current.text}</span>
          <div className="flex gap-1 ml-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-legal-blue-400" style={{ animationDelay: '0ms' }} />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-legal-blue-400" style={{ animationDelay: '150ms' }} />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-legal-blue-400" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
      </div>
    </div>
  );
}
