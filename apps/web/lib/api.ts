let _tokenPromise: Promise<void> | null = null;

async function ensureToken(): Promise<void> {
  if (typeof window === 'undefined') return;
  if (localStorage.getItem('token')) return;

  if (_tokenPromise) return _tokenPromise;

  _tokenPromise = (async () => {
    try {
      const res = await fetch('/api/v1/chat/demo-token', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.token) {
          localStorage.setItem('token', data.token);
        }
      }
    } catch {
      // silently fail — pages will show empty state
    } finally {
      _tokenPromise = null;
    }
  })();

  return _tokenPromise;
}

function getAuthHeaders(): HeadersInit {
  if (typeof window === 'undefined') return {};
  const token = localStorage.getItem('token');
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export async function apiFetch(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  await ensureToken();
  const headers = { ...getAuthHeaders(), ...options.headers };
  const baseUrl =
    typeof window !== 'undefined'
      ? ''
      : process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
  return fetch(`${baseUrl}${path}`, { ...options, headers });
}

export async function apiPost<T = unknown>(
  path: string,
  body: unknown,
  options: RequestInit = {}
): Promise<T> {
  const res = await apiFetch(path, {
    ...options,
    method: 'POST',
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)
    );
  }
  return res.json() as Promise<T>;
}
