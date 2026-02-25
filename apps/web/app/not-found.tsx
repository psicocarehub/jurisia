import { FileQuestion, Home, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-md text-center">
        <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-gray-100">
          <FileQuestion className="h-8 w-8 text-gray-400" />
        </div>
        <h1 className="mb-1 text-6xl font-bold text-gray-200">404</h1>
        <h2 className="mb-2 text-xl font-semibold text-gray-900">
          Página não encontrada
        </h2>
        <p className="mb-6 text-sm text-gray-500">
          A página que você está procurando não existe ou foi movida.
        </p>
        <div className="flex justify-center gap-3">
          <a
            href="/"
            className="flex items-center gap-2 rounded-lg bg-legal-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-legal-blue-700 transition-colors"
          >
            <Home className="h-4 w-4" />
            Página inicial
          </a>
        </div>
      </div>
    </div>
  );
}
