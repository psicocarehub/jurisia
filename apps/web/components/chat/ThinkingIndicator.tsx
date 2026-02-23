'use client';

import { cn } from '@/lib/utils';

interface ThinkingIndicatorProps {
  className?: string;
}

export function ThinkingIndicator({ className }: ThinkingIndicatorProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-2xl bg-white px-4 py-3 shadow-sm',
        className
      )}
    >
      <div className="flex gap-1">
        <span
          className="h-2 w-2 animate-bounce rounded-full bg-legal-blue-500"
          style={{ animationDelay: '0ms' }}
        />
        <span
          className="h-2 w-2 animate-bounce rounded-full bg-legal-blue-500"
          style={{ animationDelay: '150ms' }}
        />
        <span
          className="h-2 w-2 animate-bounce rounded-full bg-legal-blue-500"
          style={{ animationDelay: '300ms' }}
        />
      </div>
      <span className="text-sm text-gray-600">Pensando...</span>
    </div>
  );
}
