'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  useEditor,
  EditorContent,
  type Extensions,
  Mark,
  mergeAttributes,
} from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import {
  Bold,
  Italic,
  Heading1,
  Heading2,
  List,
  ListOrdered,
  Quote,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { LegalToolbar, type PetitionType, type CitationStatus } from './LegalToolbar';
import type { Citation } from './CitationPlugin';

const CitationHighlight = Mark.create({
  name: 'citationHighlight',

  addAttributes() {
    return {
      type: {
        default: 'legislacao',
        parseHTML: (el) => el.getAttribute('data-citation-type') || 'legislacao',
        renderHTML: (attrs) => ({ 'data-citation-type': attrs.type }),
      },
      status: {
        default: 'unchecked',
        parseHTML: (el) => el.getAttribute('data-citation-status') || 'unchecked',
        renderHTML: (attrs) => ({ 'data-citation-status': attrs.status }),
      },
    };
  },

  parseHTML() {
    return [{ tag: 'mark[data-citation]' }];
  },

  renderHTML({ HTMLAttributes }) {
    return [
      'mark',
      mergeAttributes(
        { 'data-citation': '' },
        this.options.HTMLAttributes,
        HTMLAttributes,
        { class: 'citation-highlight' }
      ),
      0,
    ];
  },
});

export interface TiptapEditorProps {
  content?: string;
  onChange?: (content: string) => void;
  isAIGenerated?: boolean;
  citations?: Citation[];
  onInsertCitation?: () => void;
  onVerifyCitations?: () => void;
  onFormatABNT?: () => void;
  onAddArticle?: () => void;
  onAddJurisprudence?: () => void;
  petitionType?: PetitionType;
  onPetitionTypeChange?: (type: PetitionType) => void;
  citationStatus?: CitationStatus;
  isVerifyingCitations?: boolean;
  className?: string;
}

const extensions: Extensions = [
  StarterKit.configure({
    heading: { levels: [1, 2, 3] },
  }),
  Placeholder.configure({
    placeholder: 'Comece a escrever sua petição...',
  }),
  CitationHighlight,
];

export function TiptapEditor({
  content = '',
  onChange,
  isAIGenerated = false,
  citations = [],
  onInsertCitation = () => {},
  onVerifyCitations = () => {},
  onFormatABNT = () => {},
  onAddArticle = () => {},
  onAddJurisprudence = () => {},
  petitionType = 'inicial',
  onPetitionTypeChange = () => {},
  citationStatus = 'unchecked',
  isVerifyingCitations = false,
  className,
}: TiptapEditorProps) {
  const [petitionTypeState, setPetitionTypeState] =
    useState<PetitionType>(petitionType);

  const editor = useEditor({
    extensions,
    content: content || undefined,
    editorProps: {
      attributes: {
        class:
          'tiptap-editor-content min-h-[400px] px-4 py-4 focus:outline-none',
      },
    },
    onUpdate: ({ editor }) => {
      onChange?.(editor.getHTML());
    },
  });

  const setContent = useCallback(
    (html: string) => {
      editor?.commands.setContent(html, false);
    },
    [editor]
  );

  useEffect(() => {
    if (content && editor && editor.getHTML() !== content) {
      setContent(content);
    }
  }, [content, editor, setContent]);

  useEffect(() => {
    onPetitionTypeChange(petitionTypeState);
  }, [petitionTypeState, onPetitionTypeChange]);

  const handlePetitionTypeChange = useCallback((type: PetitionType) => {
    setPetitionTypeState(type);
  }, []);

  const toggleBold = useCallback(() => editor?.chain().focus().toggleBold().run(), [editor]);
  const toggleItalic = useCallback(
    () => editor?.chain().focus().toggleItalic().run(),
    [editor]
  );
  const setHeading1 = useCallback(
    () => editor?.chain().focus().toggleHeading({ level: 1 }).run(),
    [editor]
  );
  const setHeading2 = useCallback(
    () => editor?.chain().focus().toggleHeading({ level: 2 }).run(),
    [editor]
  );
  const toggleBulletList = useCallback(
    () => editor?.chain().focus().toggleBulletList().run(),
    [editor]
  );
  const toggleOrderedList = useCallback(
    () => editor?.chain().focus().toggleOrderedList().run(),
    [editor]
  );
  const toggleBlockquote = useCallback(
    () => editor?.chain().focus().toggleBlockquote().run(),
    [editor]
  );

  const addArticleTemplate = useCallback(() => {
    editor
      ?.chain()
      .focus()
      .insertContent(
        '<p><strong>Art. 1º</strong> </p><p></p>'
      )
      .run();
  }, [editor]);

  const addJurisprudenceTemplate = useCallback(() => {
    editor
      ?.chain()
      .focus()
      .insertContent(
        '<p><em>Jurisprudência:</em></p><p></p>'
      )
      .run();
  }, [editor]);

  if (!editor) {
    return (
      <div className="flex min-h-[400px] items-center justify-center text-gray-500">
        Carregando editor...
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col overflow-hidden rounded-lg border border-gray-200 bg-white', className)}>
      {/* AI Disclaimer Banner */}
      {isAIGenerated && (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-2">
          <p className="text-sm text-amber-800">
            <strong>Aviso:</strong> Este conteúdo foi gerado com assistência de
            IA. Verifique todas as informações e citações antes de submeter.
          </p>
        </div>
      )}

      {/* Legal Toolbar */}
      <LegalToolbar
        petitionType={petitionTypeState}
        onPetitionTypeChange={handlePetitionTypeChange}
        onInsertCitation={onInsertCitation}
        onVerifyCitations={onVerifyCitations}
        onFormatABNT={onFormatABNT}
        onAddArticle={() => {
          addArticleTemplate();
          onAddArticle();
        }}
        onAddJurisprudence={() => {
          addJurisprudenceTemplate();
          onAddJurisprudence();
        }}
        citationStatus={citationStatus}
        isVerifying={isVerifyingCitations}
      />

      {/* Formatting Toolbar */}
      <div className="flex flex-wrap items-center gap-0.5 border-b border-gray-200 bg-white px-2 py-1">
        <FormatButton
          onClick={toggleBold}
          active={editor.isActive('bold')}
          ariaLabel="Negrito"
        >
          <Bold className="h-4 w-4" />
        </FormatButton>
        <FormatButton
          onClick={toggleItalic}
          active={editor.isActive('italic')}
          ariaLabel="Itálico"
        >
          <Italic className="h-4 w-4" />
        </FormatButton>
        <div className="mx-1 w-px self-stretch bg-gray-200" />
        <FormatButton
          onClick={setHeading1}
          active={editor.isActive('heading', { level: 1 })}
          ariaLabel="Título 1"
        >
          <Heading1 className="h-4 w-4" />
        </FormatButton>
        <FormatButton
          onClick={setHeading2}
          active={editor.isActive('heading', { level: 2 })}
          ariaLabel="Título 2"
        >
          <Heading2 className="h-4 w-4" />
        </FormatButton>
        <div className="mx-1 w-px self-stretch bg-gray-200" />
        <FormatButton
          onClick={toggleBulletList}
          active={editor.isActive('bulletList')}
          ariaLabel="Lista"
        >
          <List className="h-4 w-4" />
        </FormatButton>
        <FormatButton
          onClick={toggleOrderedList}
          active={editor.isActive('orderedList')}
          ariaLabel="Lista numerada"
        >
          <ListOrdered className="h-4 w-4" />
        </FormatButton>
        <FormatButton
          onClick={toggleBlockquote}
          active={editor.isActive('blockquote')}
          ariaLabel="Citação"
        >
          <Quote className="h-4 w-4" />
        </FormatButton>
      </div>

      {/* Editor content */}
      <EditorContent editor={editor} />
    </div>
  );
}

interface FormatButtonProps {
  onClick: () => void;
  active?: boolean;
  ariaLabel: string;
  children: React.ReactNode;
}

function FormatButton({
  onClick,
  active,
  ariaLabel,
  children,
}: FormatButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      className={cn(
        'rounded p-2 text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-900',
        active && 'bg-legal-blue-50 text-legal-blue-700'
      )}
    >
      {children}
    </button>
  );
}
