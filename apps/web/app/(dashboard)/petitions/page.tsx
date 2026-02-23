import Link from 'next/link';
import { FilePlus } from 'lucide-react';

export default function PetitionsPage() {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold text-gray-900">Petições</h1>
      <p className="mt-2 text-gray-600">
        Crie e edite petições com assistência de IA.
      </p>
      <div className="mt-8">
        <Link
          href="/petitions/novo/editor"
          className="btn-primary inline-flex items-center gap-2"
        >
          <FilePlus className="h-4 w-4" />
          Nova petição
        </Link>
      </div>
    </div>
  );
}
