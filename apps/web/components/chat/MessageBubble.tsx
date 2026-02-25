'use client';

import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Scale, User } from 'lucide-react';

interface MessageBubbleProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  className?: string;
}

export function MessageBubble({ role, content, className }: MessageBubbleProps) {
  const isUser = role === 'user';

  if (isUser) {
    return (
      <div className={cn('flex items-start gap-3 justify-end', className)}>
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-legal-blue-600 px-4 py-3 text-white shadow-sm">
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{content}</p>
        </div>
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-legal-blue-100">
          <User className="h-4 w-4 text-legal-blue-700" />
        </div>
      </div>
    );
  }

  return (
    <div className={cn('flex items-start gap-3', className)}>
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-legal-blue-500 to-legal-blue-700 shadow-sm">
        <Scale className="h-4 w-4 text-white" />
      </div>
      <div className="max-w-[85%] space-y-0">
        <div className="rounded-2xl rounded-tl-sm bg-white px-5 py-4 shadow-sm ring-1 ring-gray-100">
          <div className="prose-juris">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}
