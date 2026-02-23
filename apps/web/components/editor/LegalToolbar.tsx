'use client';

import {
  FilePlus,
  Scale,
  BookOpen,
  FileText,
  CheckCircle2,
  ChevronDown,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export type PetitionType =
  | 'inicial'
  | 'contestacao'
  | 'recurso'
  | 'agravo'
  | 'embargos'
  | 'manifestacao'
  | 'outros';

const PETITION_TYPES: { value: PetitionType; label: string }[] = [
  { value: 'inicial', label: 'Petição Inicial' },
  { value: 'contestacao', label: 'Contestação' },
  { value: 'recurso', label: 'Recurso' },
  { value: 'agravo', label: 'Agravo' },
  { value: 'embargos', label: 'Embargos' },
  { value: 'manifestacao', label: 'Manifestação' },
  { value: 'outros', label: 'Outros' },
];

export type CitationStatus = 'all_verified' | 'has_issues' | 'unchecked';

export interface LegalToolbarProps {
  petitionType: PetitionType;
  onPetitionTypeChange: (type: PetitionType) => void;
  onInsertCitation: () => void;
  onVerifyCitations: () => void;
  onFormatABNT: () => void;
  onAddArticle: () => void;
  onAddJurisprudence: () => void;
  citationStatus?: CitationStatus;
  isVerifying?: boolean;
  className?: string;
}

export function LegalToolbar({
  petitionType,
  onPetitionTypeChange,
  onInsertCitation,
  onVerifyCitations,
  onFormatABNT,
  onAddArticle,
  onAddJurisprudence,
  citationStatus = 'unchecked',
  isVerifying = false,
  className,
}: LegalToolbarProps) {
  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-2 border-b border-gray-200 bg-gray-50 px-3 py-2',
        className
      )}
    >
      {/* Tipo de petição */}
      <div className="relative">
        <select
          value={petitionType}
          onChange={(e) =>
            onPetitionTypeChange(e.target.value as PetitionType)
          }
          className="flex h-9 cursor-pointer appearance-none items-center gap-2 rounded-lg border border-gray-300 bg-white pl-3 pr-8 text-sm font-medium text-gray-700 transition-colors hover:border-gray-400 focus:border-legal-blue-500 focus:outline-none focus:ring-1 focus:ring-legal-blue-500"
        >
          {PETITION_TYPES.map(({ value, label }) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
      </div>

      <div className="h-5 w-px bg-gray-300" />

      {/* Ações jurídicas */}
      <div className="flex flex-wrap items-center gap-1">
        <ToolbarButton
          onClick={onInsertCitation}
          icon={FilePlus}
          label="Inserir Citação"
        />
        <ToolbarButton
          onClick={onVerifyCitations}
          icon={CheckCircle2}
          label="Verificar Citações"
          disabled={isVerifying}
          status={citationStatus}
        />
        <ToolbarButton
          onClick={onFormatABNT}
          icon={BookOpen}
          label="Formatar ABNT"
        />
        <ToolbarButton
          onClick={onAddArticle}
          icon={FileText}
          label="Adicionar Artigo"
        />
        <ToolbarButton
          onClick={onAddJurisprudence}
          icon={Scale}
          label="Adicionar Jurisprudência"
        />
      </div>
    </div>
  );
}

interface ToolbarButtonProps {
  onClick: () => void;
  icon: React.ElementType;
  label: string;
  disabled?: boolean;
  status?: CitationStatus;
}

function ToolbarButton({
  onClick,
  icon: Icon,
  label,
  disabled = false,
  status,
}: ToolbarButtonProps) {
  const statusColors = {
    all_verified: 'text-emerald-600',
    has_issues: 'text-amber-600',
    unchecked: '',
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className={cn(
        'inline-flex h-9 items-center gap-1.5 rounded-lg px-2.5 text-sm font-medium text-gray-700 transition-colors',
        'hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-legal-blue-500 focus:ring-offset-1',
        'disabled:cursor-not-allowed disabled:opacity-50',
        status && status !== 'unchecked' && statusColors[status]
      )}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}
