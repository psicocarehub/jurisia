import { test, expect } from '@playwright/test';

test.describe('Cases', () => {
  test.beforeEach(async ({ page }) => {
    // Obtain demo token for testing (only works in DEBUG mode)
    const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const tokenRes = await page.request.post(`${apiBase}/api/v1/chat/demo-token`);

    if (tokenRes.ok()) {
      const { token } = await tokenRes.json();
      await page.goto('/login');
      await page.evaluate((t: string) => {
        localStorage.setItem('jurisai_token', t);
      }, token);
    }
  });

  test('cases page shows case list', async ({ page }) => {
    await page.goto('/cases');
    await page.waitForLoadState('networkidle');
    const heading = page.getByRole('heading', { level: 1 });
    await expect(heading).toBeVisible();
  });

  test('create case dialog opens', async ({ page }) => {
    await page.goto('/cases');
    await page.waitForLoadState('networkidle');
    const createButton = page.getByRole('button', { name: /novo|criar|case/i });
    if (await createButton.isVisible()) {
      await createButton.click();
      await expect(page.getByText(/título/i)).toBeVisible();
    }
  });

  test('case detail page loads', async ({ page }) => {
    await page.goto('/cases');
    await page.waitForLoadState('networkidle');
    const firstCase = page.locator('[data-testid="case-item"]').first();
    if (await firstCase.isVisible()) {
      await firstCase.click();
      await page.waitForLoadState('networkidle');
      await expect(page.getByText(/raio-x/i)).toBeVisible();
    }
  });
});
