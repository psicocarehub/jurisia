'use client';

import { useChat } from 'ai/react';
import { Send } from 'lucide-react';
import { MessageBubble } from './MessageBubble';
import { ThinkingIndicator } from './ThinkingIndicator';
import { SourceCitation, type Source } from './SourceCitation';
import { cn } from '@/lib/utils';

export function ChatInterface() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
    headers: () => {
      if (typeof window === 'undefined') return {};
      const token = localStorage.getItem('token');
      return token ? { Authorization: `Bearer ${token}` } : {};
    },
  });

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <h1 className="text-lg font-semibold text-gray-900">Chat Jurídico</h1>
        <p className="text-sm text-gray-600">
          Faça perguntas sobre jurisprudência, legislação e casos.
        </p>
      </div>

      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <p className="text-gray-600">
                Como posso ajudar? Pergunte sobre leis, jurisprudência ou peça apoio na análise de documentos.
              </p>
              <p className="mt-4 text-xs text-gray-400">
                As respostas são geradas por IA e devem ser verificadas por um advogado.
              </p>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={cn(
                    'flex',
                    m.role === 'user' ? 'justify-end' : 'justify-start'
                  )}
                >
                  <div className="space-y-2">
                    <MessageBubble role={m.role as 'user' | 'assistant'} content={m.content} />
                    {m.role === 'assistant' && m.content && (
                      <SourceCitation sources={extractSources(m)} />
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <ThinkingIndicator />
                </div>
              )}
            </div>
          )}
        </div>

        <div className="border-t border-gray-200 bg-white p-4">
          <form onSubmit={handleSubmit} className="mx-auto flex max-w-3xl gap-3">
            <input
              value={input}
              onChange={handleInputChange}
              placeholder="Digite sua pergunta..."
              className="input-field flex-1"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="btn-primary flex items-center gap-2"
            >
              <Send className="h-4 w-4" />
              Enviar
            </button>
          </form>
          <p className="mx-auto mt-2 max-w-3xl text-center text-xs text-gray-400">
            A IA pode cometer erros. Consulte sempre um advogado para decisões jurídicas.
          </p>
        </div>
      </div>
    </div>
  );
}

function extractSources(message: { content?: string; toolInvocations?: unknown[] }): Source[] {
  const sources: Source[] = [];

  if (message.toolInvocations) {
    for (const tool of message.toolInvocations as Array<{ result?: { sources?: Source[] } }>) {
      if (tool.result?.sources) {
        sources.push(...tool.result.sources);
      }
    }
  }

  return sources;
}
