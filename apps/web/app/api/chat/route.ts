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

    const response = await fetch(`${API_URL}/api/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        messages: body.messages,
        stream: true,
        use_rag: true,
        use_memory: false,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      return new Response(text, { status: response.status });
    }

    const reader = response.body?.getReader();
    if (!reader) {
      return new Response('No response body', { status: 500 });
    }

    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    const stream = new ReadableStream({
      async start(controller) {
        let buffer = '';
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed.startsWith('data: ')) continue;
              const jsonStr = trimmed.slice(6);
              if (!jsonStr || jsonStr === '[DONE]') continue;

              try {
                const evt = JSON.parse(jsonStr);

                if (evt.type === 'token' && evt.content) {
                  const text = evt.content;
                  controller.enqueue(encoder.encode(`0:${JSON.stringify(text)}\n`));
                } else if (evt.type === 'sources' && evt.sources) {
                  const dataPayload = [{ sources: evt.sources }];
                  controller.enqueue(
                    encoder.encode(`2:${JSON.stringify(dataPayload)}\n`)
                  );
                } else if (evt.type === 'done') {
                  controller.enqueue(
                    encoder.encode(
                      `d:${JSON.stringify({ finishReason: 'stop' })}\n`
                    )
                  );
                } else if (evt.type === 'error') {
                  const errText = evt.content || 'Erro desconhecido';
                  controller.enqueue(encoder.encode(`0:${JSON.stringify(errText)}\n`));
                }
              } catch {
                // skip unparseable lines
              }
            }
          }

          controller.enqueue(
            encoder.encode(`d:${JSON.stringify({ finishReason: 'stop' })}\n`)
          );
        } catch (err) {
          console.error('Stream transform error:', err);
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'X-Vercel-AI-Data-Stream': 'v1',
      },
    });
  } catch (error) {
    console.error('Chat API error:', error);
    return new Response(
      JSON.stringify({ error: 'Erro ao conectar com o servidor' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
