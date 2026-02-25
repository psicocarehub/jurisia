'use client';

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';
import { useRouter, usePathname } from 'next/navigation';

interface User {
  id: string;
  email: string;
  name?: string;
  role: string;
  tenant_id: string;
}

interface AuthContextValue {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, password: string, tenantSlug: string) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
}

interface RegisterData {
  name: string;
  email: string;
  password: string;
  tenant_slug: string;
  oab_number?: string;
}

const AuthContext = createContext<AuthContextValue | null>(null);

const PUBLIC_ROUTES = ['/login', '/register'];

function decodeToken(token: string): User | null {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return {
      id: payload.sub,
      email: payload.email || '',
      role: payload.role || 'lawyer',
      tenant_id: payload.tenant_id,
    };
  } catch {
    return null;
  }
}

function isTokenExpired(token: string): boolean {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.exp * 1000 < Date.now();
  } catch {
    return true;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem('token');
    if (stored && !isTokenExpired(stored)) {
      setToken(stored);
      setUser(decodeToken(stored));
    } else if (stored) {
      localStorage.removeItem('token');
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    if (isLoading) return;
    const isPublic = PUBLIC_ROUTES.some((r) => pathname.startsWith(r));
    if (!token && !isPublic) {
      router.replace('/login');
    }
    if (token && isPublic) {
      router.replace('/chat');
    }
  }, [token, pathname, isLoading, router]);

  const login = useCallback(
    async (email: string, password: string, tenantSlug: string) => {
      const res = await fetch('/api/v1/admin/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, tenant_slug: tenantSlug }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Credenciais inválidas');
      }

      const data = await res.json();
      localStorage.setItem('token', data.access_token);
      setToken(data.access_token);
      setUser(decodeToken(data.access_token));
      router.push('/chat');
    },
    [router],
  );

  const register = useCallback(
    async (data: RegisterData) => {
      const res = await fetch('/api/v1/admin/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Falha no cadastro');
      }

      const resp = await res.json();
      localStorage.setItem('token', resp.access_token);
      setToken(resp.access_token);
      setUser(decodeToken(resp.access_token));
      router.push('/chat');
    },
    [router],
  );

  const logout = useCallback(() => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    router.push('/login');
  }, [router]);

  return (
    <AuthContext.Provider value={{ user, token, isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
