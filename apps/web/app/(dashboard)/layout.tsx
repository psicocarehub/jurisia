'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  MessageSquare,
  Briefcase,
  FileText,
  FileSignature,
  BarChart3,
  Settings,
  Bell,
  Shield,
  Newspaper,
  LogOut,
  ChevronLeft,
  Menu,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiFetch } from '@/lib/api';
import { useAuth } from '@/lib/auth-context';

const navItems = [
  { href: '/chat', label: 'Chat', icon: MessageSquare },
  { href: '/cases', label: 'Casos', icon: Briefcase },
  { href: '/documents', label: 'Documentos', icon: FileText },
  { href: '/petitions', label: 'Peticoes', icon: FileSignature },
  { href: '/jurimetrics', label: 'Jurimetria', icon: BarChart3 },
  { href: '/updates', label: 'Novidades', icon: Newspaper, badgeKey: 'updates' },
  { href: '/alerts', label: 'Alertas', icon: Bell, badgeKey: 'alerts' },
  { href: '/compliance', label: 'Compliance', icon: Shield },
  { href: '/settings', label: 'Configuracoes', icon: Settings },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const { user, logout, isLoading } = useAuth();
  const [badges, setBadges] = useState<Record<string, number>>({});
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const fetchBadges = async () => {
      try {
        const today = new Date().toISOString().split('T')[0];
        const [alertsRes, updatesRes] = await Promise.all([
          apiFetch('/api/v1/alerts?unread_only=true&limit=1').catch(() => null),
          apiFetch(`/api/v1/updates/stats?date_from=${today}&date_to=${today}`).catch(() => null),
        ]);

        const newBadges: Record<string, number> = {};
        if (alertsRes?.ok) {
          const data = await alertsRes.json();
          const count = data.total ?? (Array.isArray(data.alerts) ? data.alerts.length : 0);
          if (count > 0) newBadges.alerts = count;
        }
        if (updatesRes?.ok) {
          const data = await updatesRes.json();
          if (data.total > 0) newBadges.updates = data.total;
        }
        setBadges(newBadges);
      } catch {
        // silently ignore badge fetch errors
      }
    };

    fetchBadges();
    const interval = setInterval(fetchBadges, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-legal-blue-600 border-t-transparent" />
      </div>
    );
  }

  const sidebarContent = (
    <>
      <div className={cn(
        'flex h-16 items-center border-b border-gray-200 px-4',
        collapsed ? 'justify-center' : 'justify-between',
      )}>
        {!collapsed && (
          <Link href="/chat" className="text-lg font-semibold text-legal-blue-700">
            Juris.AI
          </Link>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="hidden lg:flex h-8 w-8 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-600"
        >
          <ChevronLeft className={cn('h-4 w-4 transition-transform', collapsed && 'rotate-180')} />
        </button>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        {navItems.map(({ href, label, icon: Icon, badgeKey }) => {
          const isActive = pathname === href || pathname.startsWith(`${href}/`);
          const badgeCount = badgeKey ? badges[badgeKey] : undefined;
          return (
            <Link
              key={href}
              href={href}
              title={collapsed ? label : undefined}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                collapsed && 'justify-center px-2',
                isActive
                  ? 'bg-legal-blue-50 text-legal-blue-700'
                  : 'text-gray-700 hover:bg-gray-100',
              )}
            >
              <Icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span className="flex-1">{label}</span>}
              {!collapsed && badgeCount !== undefined && badgeCount > 0 && (
                <span className="flex h-5 min-w-[20px] items-center justify-center rounded-full bg-red-500 px-1.5 text-[10px] font-bold text-white">
                  {badgeCount > 99 ? '99+' : badgeCount}
                </span>
              )}
              {collapsed && badgeCount !== undefined && badgeCount > 0 && (
                <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-red-500" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* User section */}
      <div className={cn(
        'border-t border-gray-200 p-3',
        collapsed && 'flex flex-col items-center',
      )}>
        {user && !collapsed && (
          <div className="mb-2 rounded-lg bg-gray-50 px-3 py-2">
            <p className="truncate text-sm font-medium text-gray-900">
              {user.email}
            </p>
            <p className="text-xs capitalize text-gray-500">{user.role}</p>
          </div>
        )}
        <button
          onClick={logout}
          title="Sair"
          className={cn(
            'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-red-50 hover:text-red-600 w-full',
            collapsed && 'justify-center px-2',
          )}
        >
          <LogOut className="h-5 w-5 shrink-0" />
          {!collapsed && <span>Sair</span>}
        </button>
      </div>
    </>
  );

  return (
    <div className="flex min-h-screen">
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/30 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile sidebar */}
      <aside className={cn(
        'fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-gray-200 bg-white transition-transform lg:hidden',
        mobileOpen ? 'translate-x-0' : '-translate-x-full',
      )}>
        {sidebarContent}
      </aside>

      {/* Desktop sidebar */}
      <aside className={cn(
        'hidden lg:flex shrink-0 flex-col border-r border-gray-200 bg-white transition-all duration-200',
        collapsed ? 'w-16' : 'w-56',
      )}>
        {sidebarContent}
      </aside>

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Mobile header */}
        <div className="flex h-14 items-center gap-3 border-b border-gray-200 bg-white px-4 lg:hidden">
          <button
            onClick={() => setMobileOpen(true)}
            className="flex h-9 w-9 items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100"
          >
            <Menu className="h-5 w-5" />
          </button>
          <span className="text-lg font-semibold text-legal-blue-700">Juris.AI</span>
        </div>

        <main className="flex-1 overflow-auto bg-gray-50">
          {children}
        </main>
      </div>
    </div>
  );
}
