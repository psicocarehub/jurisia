'use client';

import { cn } from '@/lib/utils';

interface MessageBubbleProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  className?: string;
}

export function MessageBubble({ role, content, className }: MessageBubbleProps) {
  const isUser = role === 'user';

  return (
    <div
      className={cn(
        'max-w-[85%] rounded-2xl px-4 py-3',
        isUser
          ? 'ml-auto bg-legal-blue-600 text-white'
          : 'mr-auto bg-white text-gray-900 shadow-sm',
        className
      )}
    >
      <p className="whitespace-pre-wrap text-sm leading-relaxed">{content}</p>
    </div>
  );
}
