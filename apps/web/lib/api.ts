const getAuthHeaders = (): HeadersInit => {
  if (typeof window === 'undefined') return {};
  const token = localStorage.getItem('token');
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
};

export async function apiFetch(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
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
