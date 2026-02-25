import { test, expect } from '@playwright/test';

test.describe('Smoke Tests', () => {
  test('login page loads', async ({ page }) => {
    await page.goto('/login');
    await expect(page).toHaveTitle(/Juris/i);
  });

  test('unauthenticated user is redirected to login', async ({ page }) => {
    await page.goto('/chat');
    await page.waitForURL(/login/);
    expect(page.url()).toContain('login');
  });
});
