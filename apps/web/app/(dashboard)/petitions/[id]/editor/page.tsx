'use client';

import { useCallback, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Save, FileDown } from 'lucide-react';
import { TiptapEditor } from '@/components/editor/TiptapEditor';
import { CitationPlugin } from '@/components/editor/CitationPlugin';
import type { Citation, CitationType, CitationStatus } from '@/components/editor/CitationPlugin';
import type { PetitionType, CitationStatus as ToolbarCitationStatus } from '@/components/editor/LegalToolbar';

export default function PetitionEditorPage() {
  const params = useParams();
  const petitionId = params.id as string;

  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [petitionType, setPetitionType] = useState<PetitionType>('inicial');
  const [caseNumber, setCaseNumber] = useState('');
  const [status, setStatus] = useState('rascunho');
  const [isAIGenerated, setIsAIGenerated] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const handleVerifyCitations = useCallback(() => {
    setIsVerifying(true);
    // Simular verificação - em produção, chamar API
    setTimeout(() => {
      setCitations((prev) =>
        prev.map((c) => ({
          ...c,
          status: (['verified', 'not_found', 'unchecked'] as CitationStatus[])[
            Math.floor(Math.random() * 3)
          ],
        }))
      );
      setIsVerifying(false);
    }, 1500);
  }, []);

  const citationStatus: ToolbarCitationStatus =
    citations.length === 0
      ? 'unchecked'
      : citations.every((c) => c.status === 'verified')
        ? 'all_verified'
        : citations.some(
            (c) =>
              c.status === 'not_found' ||
              c.status === 'revoked' ||
              c.status === 'outdated'
          )
          ? 'has_issues'
          : 'unchecked';

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    try {
      // TODO: chamar API para salvar
      await new Promise((r) => setTimeout(r, 800));
    } finally {
      setIsSaving(false);
    }
  }, [content, title, petitionType, caseNumber, status]);

  const handleExport = useCallback(() => {
    // Exportar como .doc ou PDF - placeholder
    const blob = new Blob([content], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title || petitionId || 'peticao'}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }, [content, title, petitionId]);

  return (
    <div className="flex h-[calc(100vh-0px)] flex-col">
      {/* Header */}
      <header className="flex shrink-0 items-center justify-between border-b border-gray-200 bg-white px-6 py-4">
        <div className="flex items-center gap-4">
          <Link
            href="/petitions"
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
          >
            <ArrowLeft className="h-4 w-4" />
            Voltar
          </Link>
          <div className="h-6 w-px bg-gray-200" />
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Título da petição"
            className="input-field max-w-md border-0 bg-transparent text-lg font-semibold focus:ring-0"
          />
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleSave}
            disabled={isSaving}
            className="btn-primary flex items-center gap-2"
          >
            <Save className="h-4 w-4" />
            {isSaving ? 'Salvando...' : 'Salvar'}
          </button>
          <button
            type="button"
            onClick={handleExport}
            className="btn-secondary flex items-center gap-2"
          >
            <FileDown className="h-4 w-4" />
            Exportar
          </button>
        </div>
      </header>

      {/* Metadata bar */}
      <div className="flex shrink-0 items-center gap-6 border-b border-gray-200 bg-white px-6 py-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-500">Tipo:</span>
          <span className="text-sm text-gray-900 capitalize">{petitionType}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-500">Processo:</span>
          <input
            type="text"
            value={caseNumber}
            onChange={(e) => setCaseNumber(e.target.value)}
            placeholder="Nº do processo"
            className="input-field max-w-40 py-1.5 text-sm"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-500">Status:</span>
          <span className="text-sm text-gray-900 capitalize">{status}</span>
        </div>
        {isAIGenerated && (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
            Conteúdo com IA
          </span>
        )}
      </div>

      {/* AI Label compliance disclaimer */}
      {isAIGenerated && (
        <div className="border-b border-amber-100 bg-amber-50/50 px-6 py-2">
          <p className="text-xs text-amber-800">
            Conformidade com rotulagem de IA: Este documento utiliza conteúdo
            gerado ou assistido por inteligência artificial.
          </p>
        </div>
      )}

      {/* Main content: Editor + Citation sidebar */}
      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-auto p-6">
          <TiptapEditor
            content={content}
            onChange={setContent}
            isAIGenerated={isAIGenerated}
            citations={citations}
            onInsertCitation={() => {
              setCitations((prev) => [
                ...prev,
                {
                  id: `citation-${Date.now()}`,
                  text: '[Inserir citação]',
                  type: 'legislacao',
                  status: 'unchecked',
                },
              ]);
            }}
            onVerifyCitations={handleVerifyCitations}
            onFormatABNT={() => {}}
            onAddArticle={() => {
              setCitations((prev) => [
                ...prev,
                {
                  id: `art-${Date.now()}`,
                  text: 'Art. 1º',
                  type: 'legislacao' as CitationType,
                  status: 'unchecked' as CitationStatus,
                },
              ]);
            }}
            onAddJurisprudence={() => {
              setCitations((prev) => [
                ...prev,
                {
                  id: `juris-${Date.now()}`,
                  text: 'Jurisprudência',
                  type: 'jurisprudencia' as CitationType,
                  status: 'unchecked' as CitationStatus,
                },
              ]);
            }}
            petitionType={petitionType}
            onPetitionTypeChange={setPetitionType}
            citationStatus={citationStatus}
            isVerifyingCitations={isVerifying}
          />
        </div>

        <aside className="w-80 shrink-0">
          <CitationPlugin
            citations={citations}
            onVerifyAll={handleVerifyCitations}
            isVerifying={isVerifying}
          />
        </aside>
      </div>
    </div>
  );
}
