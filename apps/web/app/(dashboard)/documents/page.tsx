'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { apiFetch } from '@/lib/api';
import {
  FileText,
  Upload,
  Download,
  Trash2,
  Search,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Clock,
  X,
} from 'lucide-react';

interface Doc {
  id: string;
  title: string;
  doc_type?: string;
  source?: string;
  ocr_status: string;
  classification_label?: string;
  storage_key?: string;
  file_size?: number;
  created_at?: string;
}

const STATUS_ICONS: Record<string, React.ReactNode> = {
  pending: <Clock className="h-4 w-4 text-yellow-500" />,
  processing: <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />,
  completed: <CheckCircle2 className="h-4 w-4 text-green-500" />,
  error: <AlertCircle className="h-4 w-4 text-red-500" />,
};

const STATUS_LABELS: Record<string, string> = {
  pending: 'Pendente',
  processing: 'Processando...',
  completed: 'Concluído',
  error: 'Erro',
};

export default function DocumentsPage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [search, setSearch] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<Doc | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiFetch('/api/v1/documents');
      if (res.ok) {
        const data = await res.json();
        setDocs(data.documents || []);
      }
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        const formData = new FormData();
        formData.append('file', file);
        const token = localStorage.getItem('token');
        const res = await fetch('/api/v1/documents/upload', {
          method: 'POST',
          headers: token ? { Authorization: `Bearer ${token}` } : {},
          body: formData,
        });
        if (res.ok) {
          const newDoc = await res.json();
          setDocs((prev) => [newDoc, ...prev]);
        }
      }
    } catch {
      alert('Erro ao fazer upload');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Excluir documento?')) return;
    try {
      await apiFetch(`/api/v1/documents/${id}`, { method: 'DELETE' });
      setDocs((prev) => prev.filter((d) => d.id !== id));
      if (selectedDoc?.id === id) setSelectedDoc(null);
    } catch {
      // silently fail
    }
  };

  const handleDownload = async (doc: Doc) => {
    try {
      const res = await apiFetch(`/api/v1/documents/${doc.id}/download`);
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = doc.title;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch {
      // silently fail
    }
  };

  const filtered = docs.filter((d) => {
    if (!search) return true;
    return d.title.toLowerCase().includes(search.toLowerCase());
  });

  const formatSize = (bytes?: number) => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="flex h-full">
      <div className={`${selectedDoc ? 'w-1/2 border-r' : 'w-full'} flex flex-col bg-white`}>
        <div className="border-b p-4">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <FileText className="h-5 w-5 text-legal-blue-600" />
              Documentos
            </h1>
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="btn-primary flex items-center gap-1.5 text-sm"
            >
              {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              {uploading ? 'Enviando...' : 'Upload'}
            </button>
            <input
              ref={fileRef}
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.txt,.odt,.rtf"
              className="hidden"
              onChange={(e) => handleUpload(e.target.files)}
            />
          </div>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Buscar documentos..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="input-field pl-9 text-sm"
            />
          </div>
        </div>

        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleUpload(e.dataTransfer.files); }}
          className={`flex-1 overflow-auto ${dragOver ? 'bg-legal-blue-50 border-2 border-dashed border-legal-blue-400' : ''}`}
        >
          {dragOver && (
            <div className="flex items-center justify-center h-40 text-legal-blue-600 font-medium">
              <Upload className="h-8 w-8 mr-2" />
              Solte os arquivos aqui
            </div>
          )}

          {loading ? (
            <div className="p-8 text-center text-gray-400">Carregando...</div>
          ) : filtered.length === 0 ? (
            <div className="p-12 text-center">
              <FileText className="h-12 w-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">Nenhum documento encontrado</p>
              <p className="text-gray-400 text-sm mt-1">Arraste arquivos ou clique em Upload</p>
            </div>
          ) : (
            <div className="divide-y">
              {filtered.map((doc) => (
                <div
                  key={doc.id}
                  onClick={() => setSelectedDoc(doc)}
                  className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors flex items-center gap-3 ${
                    selectedDoc?.id === doc.id ? 'bg-legal-blue-50' : ''
                  }`}
                >
                  <FileText className="h-8 w-8 text-gray-400 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-gray-900 text-sm truncate">{doc.title}</h3>
                    <div className="flex gap-3 mt-1 text-xs text-gray-500 items-center">
                      <span className="flex items-center gap-1">
                        {STATUS_ICONS[doc.ocr_status] || STATUS_ICONS.pending}
                        {STATUS_LABELS[doc.ocr_status] || doc.ocr_status}
                      </span>
                      {doc.classification_label && (
                        <span className="px-1.5 py-0.5 bg-purple-50 text-purple-700 rounded">
                          {doc.classification_label.replace(/_/g, ' ')}
                        </span>
                      )}
                      {doc.file_size && <span>{formatSize(doc.file_size)}</span>}
                    </div>
                  </div>
                  <div className="flex gap-1 shrink-0">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDownload(doc); }}
                      className="p-1.5 text-gray-400 hover:text-legal-blue-600 rounded"
                    >
                      <Download className="h-4 w-4" />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(doc.id); }}
                      className="p-1.5 text-gray-400 hover:text-red-500 rounded"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Detail panel */}
      {selectedDoc && (
        <div className="w-1/2 flex flex-col bg-white">
          <div className="border-b p-4 flex justify-between items-center">
            <h2 className="font-semibold text-gray-900 text-sm truncate">{selectedDoc.title}</h2>
            <button onClick={() => setSelectedDoc(null)} className="text-gray-400 hover:text-gray-600">
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex-1 overflow-auto p-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <dt className="text-xs font-medium text-gray-500 uppercase">Status OCR</dt>
                <dd className="mt-1 text-sm flex items-center gap-1">
                  {STATUS_ICONS[selectedDoc.ocr_status]}
                  {STATUS_LABELS[selectedDoc.ocr_status] || selectedDoc.ocr_status}
                </dd>
              </div>
              {selectedDoc.classification_label && (
                <div>
                  <dt className="text-xs font-medium text-gray-500 uppercase">Classificação</dt>
                  <dd className="mt-1 text-sm">{selectedDoc.classification_label.replace(/_/g, ' ')}</dd>
                </div>
              )}
              {selectedDoc.file_size && (
                <div>
                  <dt className="text-xs font-medium text-gray-500 uppercase">Tamanho</dt>
                  <dd className="mt-1 text-sm">{formatSize(selectedDoc.file_size)}</dd>
                </div>
              )}
              {selectedDoc.created_at && (
                <div>
                  <dt className="text-xs font-medium text-gray-500 uppercase">Upload em</dt>
                  <dd className="mt-1 text-sm">{new Date(selectedDoc.created_at).toLocaleString('pt-BR')}</dd>
                </div>
              )}
              {selectedDoc.source && (
                <div>
                  <dt className="text-xs font-medium text-gray-500 uppercase">Fonte</dt>
                  <dd className="mt-1 text-sm">{selectedDoc.source}</dd>
                </div>
              )}
            </div>
            <div className="flex gap-2 pt-4">
              <button
                onClick={() => handleDownload(selectedDoc)}
                className="btn-primary text-sm flex items-center gap-1.5"
              >
                <Download className="h-4 w-4" />
                Download
              </button>
              <button
                onClick={() => handleDelete(selectedDoc.id)}
                className="btn-secondary text-sm flex items-center gap-1.5 text-red-600 hover:text-red-700"
              >
                <Trash2 className="h-4 w-4" />
                Excluir
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
