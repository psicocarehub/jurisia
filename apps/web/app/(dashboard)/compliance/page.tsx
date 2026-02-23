'use client';

import { useState } from 'react';
import { apiFetch } from '@/lib/api';
import {
  Shield,
  Search,
  AlertTriangle,
  CheckCircle2,
  Building2,
  Loader2,
} from 'lucide-react';

interface Sanction {
  source: string;
  entity_name: string;
  identifier: string;
  sanction_type: string;
  start_date: string;
  end_date: string;
  reason: string;
  organ: string;
}

interface ComplianceResult {
  identifier: string;
  sanctions: Sanction[];
  total: number;
  clean: boolean;
}

interface CNPJResult {
  cnpj: string;
  razao_social: string;
  nome_fantasia: string;
  situacao_cadastral: string;
  data_abertura: string;
  cnae_principal: string;
  endereco: string;
  porte: string;
}

export default function CompliancePage() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [complianceResult, setComplianceResult] = useState<ComplianceResult | null>(null);
  const [cnpjResult, setCnpjResult] = useState<CNPJResult | null>(null);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    setComplianceResult(null);
    setCnpjResult(null);

    const clean = query.replace(/[.\-/]/g, '');

    try {
      // Check sanctions
      const sanctionRes = await apiFetch(`/api/v1/compliance/check/${clean}`);
      if (sanctionRes.ok) {
        const data = await sanctionRes.json();
        setComplianceResult(data);
      }

      // If CNPJ (14 digits), also fetch company data
      if (clean.length === 14) {
        const cnpjRes = await apiFetch(`/api/v1/compliance/cnpj/${clean}`);
        if (cnpjRes.ok) {
          const data = await cnpjRes.json();
          setCnpjResult(data);
        }
      }
    } catch (e: any) {
      setError(e.message || 'Erro na consulta');
    } finally {
      setLoading(false);
    }
  };

  const formatCnpjCpf = (v: string) => {
    const clean = v.replace(/\D/g, '');
    if (clean.length <= 11) {
      return clean.replace(/(\d{3})(\d{3})(\d{3})(\d{2})/, '$1.$2.$3-$4');
    }
    return clean.replace(/(\d{2})(\d{3})(\d{3})(\d{4})(\d{2})/, '$1.$2.$3/$4-$5');
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="flex items-center gap-2 mb-6">
        <Shield className="h-6 w-6 text-legal-blue-600" />
        <h1 className="text-xl font-semibold text-gray-900">Compliance</h1>
      </div>

      <div className="bg-white rounded-xl border p-6 mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Verificar Sanções e Dados Cadastrais</h3>
        <p className="text-xs text-gray-500 mb-4">
          Consulta CEIS, CNEP e CEAF (Portal da Transparência) e dados de CNPJ (Receita Federal)
        </p>
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Digite o CNPJ ou CPF..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="input-field pl-9 text-sm"
            />
          </div>
          <button onClick={handleSearch} disabled={loading} className="btn-primary text-sm">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Consultar'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
          {error}
        </div>
      )}

      {complianceResult && (
        <div className="bg-white rounded-xl border p-6 mb-4">
          <div className="flex items-center gap-3 mb-4">
            {complianceResult.clean ? (
              <CheckCircle2 className="h-8 w-8 text-green-500" />
            ) : (
              <AlertTriangle className="h-8 w-8 text-red-500" />
            )}
            <div>
              <h3 className="font-semibold text-gray-900">
                {complianceResult.clean ? 'Nenhuma Sanção Encontrada' : `${complianceResult.total} Sanção(ões) Encontrada(s)`}
              </h3>
              <p className="text-sm text-gray-500">
                Identificador: {formatCnpjCpf(complianceResult.identifier)}
              </p>
            </div>
          </div>

          {complianceResult.sanctions.length > 0 && (
            <div className="space-y-3 mt-4">
              {complianceResult.sanctions.map((s, i) => (
                <div key={i} className="border rounded-lg p-4 bg-red-50/50">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-medium text-gray-900 text-sm">{s.entity_name}</h4>
                    <span className="text-xs px-2 py-0.5 bg-red-100 text-red-700 rounded">
                      {s.source}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div><span className="text-gray-500">Tipo:</span> {s.sanction_type}</div>
                    <div><span className="text-gray-500">Órgão:</span> {s.organ}</div>
                    <div><span className="text-gray-500">Início:</span> {s.start_date}</div>
                    <div><span className="text-gray-500">Fim:</span> {s.end_date || 'Vigente'}</div>
                  </div>
                  {s.reason && (
                    <p className="text-xs text-gray-600 mt-2">{s.reason}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {cnpjResult && (
        <div className="bg-white rounded-xl border p-6">
          <div className="flex items-center gap-3 mb-4">
            <Building2 className="h-6 w-6 text-legal-blue-600" />
            <h3 className="font-semibold text-gray-900">Dados Cadastrais</h3>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {[
              ['CNPJ', formatCnpjCpf(cnpjResult.cnpj)],
              ['Razão Social', cnpjResult.razao_social],
              ['Nome Fantasia', cnpjResult.nome_fantasia],
              ['Situação', cnpjResult.situacao_cadastral],
              ['Data de Abertura', cnpjResult.data_abertura],
              ['CNAE Principal', cnpjResult.cnae_principal],
              ['Porte', cnpjResult.porte],
              ['Endereço', cnpjResult.endereco],
            ].map(([label, value]) =>
              value ? (
                <div key={label as string}>
                  <dt className="text-xs font-medium text-gray-500 uppercase">{label}</dt>
                  <dd className="mt-1 text-gray-900">{value}</dd>
                </div>
              ) : null
            )}
          </div>
        </div>
      )}
    </div>
  );
}
