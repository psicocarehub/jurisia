'use client';

import { useChat } from 'ai/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Send, Scale, MessageSquare, Trash2 } from 'lucide-react';
import { MessageBubble } from './MessageBubble';
import { ThinkingIndicator } from './ThinkingIndicator';
import { SourceCitation, type Source } from './SourceCitation';
import { SuggestedQuestions } from './SuggestedQuestions';
import { cn } from '@/lib/utils';

const STARTER_QUESTIONS = [
  'Quais são os requisitos de uma petição inicial?',
  'Como funciona o recurso de apelação no CPC?',
  'Quais os direitos do consumidor em compras online?',
  'O que diz a Súmula 331 do TST?',
  'Qual o prazo prescricional para ação de cobrança?',
  'Como funciona a usucapião extraordinária?',
];

function extractFollowUps(content: string): string[] {
  const lines = content.split('\n');
  const suggestions: string[] = [];
  let inSection = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (/perguntas?\s+(sugerida|relacionada|para\s+aprofund)/i.test(trimmed)) {
      inSection = true;
      continue;
    }
    if (inSection) {
      const match = trimmed.match(/^[-•*\d.]+\s*\*{0,2}(.+?)\*{0,2}\??$/);
      if (match) {
        const q = match[1].trim().replace(/\?$/, '') + '?';
        if (q.length > 10 && q.length < 120) suggestions.push(q);
      }
      if (suggestions.length >= 3) break;
    }
  }
  return suggestions;
}

interface MessageSources {
  [messageId: string]: Source[];
}

export function ChatInterface() {
  const [messageSources, setMessageSources] = useState<MessageSources>({});
  const [pendingSources, setPendingSources] = useState<Source[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  const getHeaders = (): Record<string, string> => {
    if (typeof window === 'undefined') return {};
    const token = localStorage.getItem('token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const { messages, input, handleInputChange, handleSubmit, isLoading, setInput, data } = useChat({
    api: '/api/chat',
    headers: getHeaders(),
    onResponse() {
      setPendingSources([]);
    },
    onFinish(message) {
      if (data && data.length > 0) {
        const lastData = data[data.length - 1] as { sources?: Source[] } | undefined;
        if (lastData?.sources) {
          setMessageSources((prev) => ({ ...prev, [message.id]: lastData.sources! }));
        }
      }
      if (pendingSources.length > 0) {
        setMessageSources((prev) => ({ ...prev, [message.id]: pendingSources }));
      }
    },
  });

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const handleSelectQuestion = useCallback(
    (q: string) => {
      setInput(q);
      setTimeout(() => {
        const form = document.querySelector('form[data-chat-form]') as HTMLFormElement;
        if (form) form.requestSubmit();
      }, 50);
    },
    [setInput]
  );

  const lastAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant');
  const followUps = lastAssistantMsg ? extractFollowUps(lastAssistantMsg.content) : [];

  return (
    <div className="flex h-full flex-col bg-gray-50">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4 shadow-sm">
        <div className="mx-auto flex max-w-3xl items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-legal-blue-500 to-legal-blue-700 shadow-sm">
            <Scale className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Juris.AI</h1>
            <p className="text-xs text-gray-500">Assistente jurídico inteligente</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center pt-12 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-legal-blue-500 to-legal-blue-700 shadow-lg">
                <Scale className="h-8 w-8 text-white" />
              </div>
              <h2 className="mt-6 text-xl font-semibold text-gray-900">
                Como posso ajudar?
              </h2>
              <p className="mt-2 max-w-md text-sm text-gray-500">
                Pergunte sobre legislação, jurisprudência, elaboração de peças ou análise de casos.
                Vou buscar nas fontes e fundamentar a resposta.
              </p>
              <div className="mt-8 grid w-full max-w-lg grid-cols-1 gap-2 sm:grid-cols-2">
                {STARTER_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => handleSelectQuestion(q)}
                    className="flex items-start gap-2 rounded-xl border border-gray-200 bg-white px-4 py-3 text-left text-sm text-gray-700 transition-all hover:border-legal-blue-300 hover:bg-legal-blue-50 hover:shadow-sm"
                  >
                    <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-legal-blue-400" />
                    <span>{q}</span>
                  </button>
                ))}
              </div>
              <p className="mt-8 text-[11px] text-gray-400">
                Conteúdo gerado com auxílio de IA — CNJ Resolução 615/2025
              </p>
            </div>
          ) : (
            <div className="space-y-5">
              {messages.map((m, idx) => {
                const isLastAssistant =
                  m.role === 'assistant' && idx === messages.length - 1;
                const msgSources = messageSources[m.id] || [];

                return (
                  <div key={m.id}>
                    <MessageBubble
                      role={m.role as 'user' | 'assistant'}
                      content={m.content}
                    />
                    {m.role === 'assistant' && msgSources.length > 0 && (
                      <SourceCitation sources={msgSources} />
                    )}
                    {isLastAssistant && !isLoading && followUps.length > 0 && (
                      <SuggestedQuestions
                        questions={followUps}
                        onSelect={handleSelectQuestion}
                      />
                    )}
                  </div>
                );
              })}
              {isLoading && <ThinkingIndicator />}
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 bg-white p-4 shadow-[0_-2px_10px_rgba(0,0,0,0.04)]">
        <form
          data-chat-form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-end gap-3"
        >
          <div className="relative flex-1">
            <textarea
              value={input}
              onChange={(e) => {
                handleInputChange(e);
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (input.trim() && !isLoading) {
                    handleSubmit(e as unknown as React.FormEvent);
                  }
                }
              }}
              placeholder="Descreva sua dúvida jurídica..."
              className="block w-full resize-none rounded-xl border border-gray-300 bg-gray-50 px-4 py-3 pr-12 text-sm placeholder-gray-400 transition-colors focus:border-legal-blue-500 focus:bg-white focus:outline-none focus:ring-1 focus:ring-legal-blue-500"
              rows={1}
              disabled={isLoading}
            />
          </div>
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className={cn(
              'flex h-11 w-11 shrink-0 items-center justify-center rounded-xl transition-all',
              input.trim() && !isLoading
                ? 'bg-legal-blue-600 text-white shadow-sm hover:bg-legal-blue-700'
                : 'bg-gray-100 text-gray-400'
            )}
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
        <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-gray-400">
          A IA pode cometer erros. Consulte sempre um advogado para decisões jurídicas.
        </p>
      </div>
    </div>
  );
}
