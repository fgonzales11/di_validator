import {defineConfig, devices} from '@playwright/test'

if (!process.env.DI_SITE_URL || !process.env.DI_SITE_TOKEN) throw new Error('Set DI_SITE_URL and DI_SITE_TOKEN for private hosted verification')

export default defineConfig({
  testDir: './cloud-e2e', timeout: 360000, expect: {timeout: 45000}, workers: 1,
  reporter: 'list', outputDir: '../runtime/cloud-deploy/browser-results',
  use: {baseURL: process.env.DI_SITE_URL, extraHTTPHeaders: {'OAI-Sites-Authorization': `Bearer ${process.env.DI_SITE_TOKEN}`}, trace: 'off'},
  projects: [{name: 'hosted', use: {...devices['Desktop Chrome'], viewport: {width: 1440, height: 1000}}}],
})
