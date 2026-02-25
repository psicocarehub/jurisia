'use client';

import { useEffect, useState, useCallback } from 'react';

const STORAGE_KEY = 'jurisai_dark_mode';

export function useDarkMode() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'true') {
      setIsDark(true);
      document.documentElement.classList.add('dark');
    } else if (stored === null) {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (prefersDark) {
        setIsDark(true);
        document.documentElement.classList.add('dark');
      }
    }
  }, []);

  const toggle = useCallback(() => {
    setIsDark((prev) => {
      const next = !prev;
      if (next) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      localStorage.setItem(STORAGE_KEY, String(next));
      return next;
    });
  }, []);

  return { isDark, toggle };
}
