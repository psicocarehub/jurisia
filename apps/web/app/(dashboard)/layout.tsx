'use client';

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
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/chat', label: 'Chat', icon: MessageSquare },
  { href: '/cases', label: 'Casos', icon: Briefcase },
  { href: '/documents', label: 'Documentos', icon: FileText },
  { href: '/petitions', label: 'Petições', icon: FileSignature },
  { href: '/jurimetrics', label: 'Jurimetria', icon: BarChart3 },
  { href: '/updates', label: 'Novidades', icon: Newspaper },
  { href: '/alerts', label: 'Alertas', icon: Bell },
  { href: '/compliance', label: 'Compliance', icon: Shield },
  { href: '/settings', label: 'Configurações', icon: Settings },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen">
      <aside className="w-56 shrink-0 border-r border-gray-200 bg-white">
        <div className="flex h-16 items-center border-b border-gray-200 px-6">
          <Link href="/chat" className="text-lg font-semibold text-legal-blue-700">
            Juris.AI
          </Link>
        </div>
        <nav className="space-y-1 p-4">
          {navItems.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-legal-blue-50 text-legal-blue-700'
                    : 'text-gray-700 hover:bg-gray-100'
                )}
              >
                <Icon className="h-5 w-5 shrink-0" />
                {label}
              </Link>
            );
          })}
        </nav>
      </aside>
      <main className="flex-1 overflow-auto bg-gray-50">
        {children}
      </main>
    </div>
  );
}
