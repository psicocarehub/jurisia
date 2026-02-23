import { NextRequest } from 'next/server';

const API_URL = process.env.API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const token = request.headers.get('authorization');

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (token) {
      headers['Authorization'] = token;
    }

    const backendUrl = `${API_URL}/api/v1/search`;
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    if (response.ok) {
      const data = await response.json();
      return Response.json(data);
    }

    // Fallback: retorna dados mock se backend não disponível
    const { query } = body;
    if (typeof query === 'string' && query.trim()) {
      return Response.json({
        results: [
          {
            id: '1',
            title: 'Lei 8.078/90 - CDC',
            court: 'STF',
            date: '1990-09-11',
            doc_type: 'legislação',
            content: 'Art. 1º O presente código estabelece normas de proteção e defesa do consumidor...',
            score: 0.95,
            highlight: 'Estabelece normas de <mark>proteção</mark> do consumidor',
          },
          {
            id: '2',
            title: 'REsp 123456 - Responsabilidade civil',
            court: 'STJ',
            date: '2023-05-15',
            doc_type: 'jurisprudência',
            content: 'Recurso especial. Responsabilidade civil. Danos morais...',
            score: 0.82,
          },
        ],
      });
    }

    return Response.json({ results: [] });
  } catch {
    return Response.json(
      { error: 'Erro ao buscar', results: [] },
      { status: 200 }
    );
  }
}
