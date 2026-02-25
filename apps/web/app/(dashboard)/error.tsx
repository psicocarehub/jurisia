'use client';

import { useEffect } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import Link from 'next/link';

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Dashboard error:', error);
  }, [error]);

  return (
    <div className="flex items-center justify-center h-full bg-gray-50 dark:bg-gray-900 p-8">
      <div className="max-w-md w-full bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-8 text-center">
        <div className="mx-auto w-14 h-14 bg-red-100 rounded-full flex items-center justify-center mb-4">
          <AlertTriangle className="h-7 w-7 text-red-600" />
        </div>

        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Algo deu errado
        </h2>

        <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
          Ocorreu um erro inesperado nesta seção. Você pode tentar recarregar
          ou voltar à página inicial.
        </p>

        {error.message && (
          <p className="text-xs text-gray-400 bg-gray-50 dark:bg-gray-700 rounded p-3 mb-6 font-mono break-all">
            {error.message}
          </p>
        )}

        <div className="flex gap-3 justify-center">
          <button
            onClick={reset}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-legal-blue-600 rounded-lg hover:bg-legal-blue-700 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Tentar novamente
          </button>

          <Link
            href="/chat"
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <Home className="h-4 w-4" />
            Início
          </Link>
        </div>

        {error.digest && (
          <p className="text-xs text-gray-400 mt-6">
            Código: {error.digest}
          </p>
        )}
      </div>
    </div>
  );
}
