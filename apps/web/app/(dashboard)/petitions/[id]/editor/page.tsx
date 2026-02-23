'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Save, FileDown, Loader2, CheckCircle } from 'lucide-react';
import { TiptapEditor } from '@/components/editor/TiptapEditor';
import { CitationPlugin } from '@/components/editor/CitationPlugin';
import type { Citation, CitationType, CitationStatus } from '@/components/editor/CitationPlugin';
import type { PetitionType, CitationStatus as ToolbarCitationStatus } from '@/components/editor/LegalToolbar';
import { apiFetch, apiPost } from '@/lib/api';

export default function PetitionEditorPage() {
  const params = useParams();
  const router = useRouter();
  const petitionId = params.id as string;
  const isNew = petitionId === 'novo';

  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [petitionType, setPetitionType] = useState<PetitionType>('inicial');
  const [caseNumber, setCaseNumber] = useState('');
  const [status, setStatus] = useState('rascunho');
  const [isAIGenerated, setIsAIGenerated] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [loading, setLoading] = useState(!isNew);
  const [savedId, setSavedId] = useState<string | null>(isNew ? null : petitionId);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [exportFormat, setExportFormat] = useState<string | null>(null);

  const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const contentRef = useRef(content);
  const titleRef = useRef(title);

  useEffect(() => {
    contentRef.current = content;
  }, [content]);
  useEffect(() => {
    titleRef.current = title;
  }, [title]);

  useEffect(() => {
    if (isNew) return;
    const loadPetition = async () => {
      try {
        const resp = await apiFetch(`/api/v1/petitions/${petitionId}`);
        if (resp.ok) {
          const data = await resp.json();
          setTitle(data.title || '');
          setContent(data.content || '');
          setPetitionType(data.petition_type || 'inicial');
          setStatus(data.status || 'draft');
          setIsAIGenerated(data.ai_generated || false);
          setSavedId(data.id);
        }
      } catch (err) {
        console.error('Failed to load petition:', err);
      } finally {
        setLoading(false);
      }
    };
    loadPetition();
  }, [petitionId, isNew]);

  const doSave = useCallback(async (showIndicator = true) => {
    if (showIndicator) setIsSaving(true);
    try {
      if (savedId) {
        const resp = await apiFetch(`/api/v1/petitions/${savedId}`, {
          method: 'PATCH',
          body: JSON.stringify({
            title: titleRef.current,
            content: contentRef.current,
            petition_type: petitionType,
            status,
          }),
        });
        if (resp.ok) {
          setLastSaved(new Date());
        }
      } else {
        const data = await apiPost<{ id: string }>('/api/v1/petitions', {
          title: titleRef.current || 'Nova Petição',
          content: contentRef.current,
          petition_type: petitionType,
        });
        setSavedId(data.id);
        setLastSaved(new Date());
        router.replace(`/petitions/${data.id}/editor`);
      }
    } catch (err) {
      console.error('Save failed:', err);
    } finally {
      if (showIndicator) setIsSaving(false);
    }
  }, [savedId, petitionType, status, router]);

  const scheduleAutoSave = useCallback(() => {
    if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current);
    autoSaveTimer.current = setTimeout(() => {
      doSave(false);
    }, 2000);
  }, [doSave]);

  const handleContentChange = useCallback((newContent: string) => {
    setContent(newContent);
    scheduleAutoSave();
  }, [scheduleAutoSave]);

  const handleTitleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setTitle(e.target.value);
    scheduleAutoSave();
  }, [scheduleAutoSave]);

  const handleVerifyCitations = useCallback(async () => {
    if (!savedId) return;
    setIsVerifying(true);
    try {
      const resp = await apiFetch(`/api/v1/petitions/${savedId}/verify-citations`, {
        method: 'POST',
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data.citations && Array.isArray(data.citations)) {
          setCitations(
            data.citations.map((c: any) => ({
              id: c.id || `cit-${Date.now()}-${Math.random()}`,
              text: c.text || c.reference || '',
              type: (c.type || 'legislacao') as CitationType,
              status: (c.status || 'unchecked') as CitationStatus,
            }))
          );
        }
      }
    } catch (err) {
      console.error('Citation verification failed:', err);
    } finally {
      setIsVerifying(false);
    }
  }, [savedId]);

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
    await doSave(true);
  }, [doSave]);

  const handleExport = useCallback(async (format: string) => {
    if (!savedId) return;
    setExportFormat(format);
    try {
      const resp = await apiFetch(`/api/v1/petitions/${savedId}/export/${format}`);
      if (resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${title || 'peticao'}.${format}`;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        const fallbackBlob = new Blob([content], { type: 'text/html' });
        const url = URL.createObjectURL(fallbackBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${title || 'peticao'}.html`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch {
      const blob = new Blob([content], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title || 'peticao'}.html`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExportFormat(null);
    }
  }, [savedId, title, content]);

  if (loading) {
    return (
      <div className="flex h-[calc(100vh-0px)] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    );
  }

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
            onChange={handleTitleChange}
            placeholder="Título da petição"
            className="input-field max-w-md border-0 bg-transparent text-lg font-semibold focus:ring-0"
          />
        </div>
        <div className="flex items-center gap-2">
          {lastSaved && (
            <span className="flex items-center gap-1 text-xs text-green-600">
              <CheckCircle className="h-3 w-3" />
              Salvo {lastSaved.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
          <button
            type="button"
            onClick={handleSave}
            disabled={isSaving}
            className="btn-primary flex items-center gap-2"
          >
            {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
            {isSaving ? 'Salvando...' : 'Salvar'}
          </button>
          <div className="relative">
            <button
              type="button"
              onClick={() => handleExport('pdf')}
              disabled={!!exportFormat}
              className="btn-secondary flex items-center gap-2"
            >
              {exportFormat ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileDown className="h-4 w-4" />}
              PDF
            </button>
          </div>
          <button
            type="button"
            onClick={() => handleExport('docx')}
            disabled={!!exportFormat}
            className="btn-secondary flex items-center gap-2"
          >
            {exportFormat === 'docx' ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileDown className="h-4 w-4" />}
            DOCX
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

      {isAIGenerated && (
        <div className="border-b border-amber-100 bg-amber-50/50 px-6 py-2">
          <p className="text-xs text-amber-800">
            Conformidade com rotulagem de IA: Este documento utiliza conteúdo
            gerado ou assistido por inteligência artificial.
          </p>
        </div>
      )}

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-auto p-6">
          <TiptapEditor
            content={content}
            onChange={handleContentChange}
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
