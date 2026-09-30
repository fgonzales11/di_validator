import { test, expect } from '@playwright/test'
import { mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

const evidence = resolve('../runtime/meter-lab/verification')
test.use({
  channel: process.env.METER_LAB_BROWSER_CHANNEL || undefined,
  launchOptions: { timeout: 90000 },
})
test('Meter Lab setup, ARM charts, stored outcomes, comparison, and evidence', async ({ page, request }) => {
  test.setTimeout(600000)
  mkdirSync(evidence, { recursive: true })
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/meter-lab')
  await expect(page.getByRole('heading', { name: 'Meter Lab', exact: true })).toBeVisible()
  await page.getByRole('combobox', { name: 'Agent', exact: true }).selectOption('fault')
  await expect(page.getByRole('combobox', { name: 'Meter form', exact: true })).toHaveValue('GENX_PP')
  await page.getByRole('combobox', { name: 'Scenario', exact: true }).selectOption('1A_val1')
  const config = page
    .locator('details')
    .filter({ has: page.locator('summary', { hasText: 'Agent configuration' }) })
  await config.locator('summary').click()
  const setting = config.locator('input').first()
  const previous = await setting.inputValue()
  await setting.fill('not-a-number')
  await page.getByRole('button', { name: 'Validate & preview', exact: true }).click()
  await expect(page.getByRole('alert')).toBeVisible()
  await setting.fill(previous)
  await config.locator('summary').click()
  await page.getByRole('button', { name: 'Validate & preview', exact: true }).click()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('img', { name: 'Scenario input preview' })).toBeVisible()
  await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 180000 })
  await expect(page.getByRole('button', { name: /Stored outcomes, waiting/ })).toBeVisible()
  console.log('Validated fault setup and preview')
  const runs = await (await request.get('/api/v1/meter-lab/runs')).json()
  const faults = runs.filter(
    (r: any) =>
      r.agent === 'fault' &&
      ['Verification: fault_regression', '1A val1'].includes(r.name) &&
      r.state === 'completed' &&
      r.manifest.total_samples === 400 &&
      r.checks.verdict === 'pass',
  )
  expect(faults.length).toBeGreaterThan(1)
  const run = faults[0]
  await page.goto('/meter-lab?run=' + run.id)
  await expect(page.getByRole('combobox', { name: 'Agent', exact: true })).toHaveValue('fault')
  await expect(page.locator('.js-plotly-plot')).toHaveCount(4, { timeout: 180000 })
  console.log('Loaded saved ARM charts')
  const series = await (await request.get('/api/v1/meter-lab/runs/' + run.id + '/series')).json()
  expect(series.source).toBe('ARM analysis diagnostics')
  expect(series.diagnostics[0].distance.estimated_distance_km).toBeCloseTo(0.8908903568462709, 6)
  await page
    .getByRole('combobox', { name: 'Compare with a second run', exact: true })
    .selectOption(faults[1].id)
  await expect(page.getByText(/Configuration matches/)).toBeVisible()
  console.log('Compared two saved runs')
  await page.getByRole('tab', { name: /^checks$/i }).click()
  await expect(page.getByText('ARM / Python parity', { exact: true })).toBeVisible()
  await page.getByRole('tab', { name: /^results$/i }).click()
  const rows = await (await request.get('/api/v1/meter-lab/runs/' + run.id + '/outcomes')).json()
  expect(rows.rows).toHaveLength(2)
  expect(new Set(rows.rows.map((r: any) => r.decoded.id)).size).toBe(1)
  for (const [label, count] of [
    ['Scenario', run.telemetry.transmitted],
    ['Replay transport', run.telemetry.received],
    ['HW 4.2 ARM agent', run.telemetry.processed],
    ['DataServer', run.telemetry.sdk_attempts],
    ['Stored outcomes', rows.rows.length],
    ['Analysis diagnostics', series.diagnostics.length],
  ]) {
    await expect(
      page.getByRole('button', { name: `${label}, completed, ${count} observed. Show details`, exact: true }),
    ).toBeVisible()
  }
  console.log('Verified checks and actual correlated DataServer rows')
  await expect(page.getByRole('heading', { name: 'DataServer-stored outcomes', exact: true })).toBeVisible()
  await page.getByRole('tab', { name: /^charts$/i }).click()
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
    .toBe(true)
  await page.evaluate(() => window.scrollTo(0, 0))
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
  await page.screenshot({
    path: resolve(evidence, 'meter-lab-' + test.info().project.name + '.png'),
    fullPage: true,
  })
  const download = await Promise.all([
    page.waitForEvent('download'),
    page
      .getByRole('link', { name: /evidence/i })
      .first()
      .click(),
  ])
  expect(download[0].suggestedFilename()).toMatch(/meter-lab-.*\.zip$/)
  await download[0].saveAs(resolve(evidence, 'browser-' + test.info().project.name + '.zip'))
  console.log('Saved screenshot and downloaded evidence')
  expect(errors).toEqual([])
})

test('scenario controls survive navigation and stop orderly', async ({ page, request }) => {
  test.skip(
    process.env.METER_LAB_CONTROLS !== '1' || test.info().project.name !== 'desktop',
    'Explicit local integration run only',
  )
  test.setTimeout(600000)
  const runs = await (await request.get('/api/v1/meter-lab/runs')).json()
  expect(runs.every((r: any) => ['completed', 'cancelled', 'failed', 'interrupted'].includes(r.state))).toBe(
    true,
  )
  await page.goto('/meter-lab')
  await page.getByRole('combobox', { name: 'Agent', exact: true }).selectOption('pv')
  await page.getByRole('combobox', { name: 'Scenario', exact: true }).selectOption('sign_changes')
  await page.getByRole('combobox', { name: 'Playback speed', exact: true }).selectOption('1')
  const created = page.waitForResponse(
    (r) => r.url().endsWith('/api/v1/meter-lab/runs') && r.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Start run', exact: true }).click()
  const run = await (await created).json()
  const detail = async () => await (await request.get('/api/v1/meter-lab/runs/' + run.id)).json()
  await expect.poll(async () => (await detail()).telemetry.processed, { timeout: 180000 }).toBeGreaterThan(2)
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  await expect.poll(async () => (await detail()).state).toBe('paused')
  const paused = await detail()
  await page.goto('/events')
  await page.goto('/meter-lab?run=' + run.id)
  await expect(page.getByRole('button', { name: 'Resume', exact: true })).toBeVisible()
  const still = await detail()
  expect(still.telemetry.processed).toBe(paused.telemetry.processed)
  expect(still.telemetry.scenario_time).toBe(paused.telemetry.scenario_time)
  await page.getByRole('button', { name: 'Resume', exact: true }).click()
  await expect
    .poll(async () => (await detail()).telemetry.processed)
    .toBeGreaterThan(paused.telemetry.processed)
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect.poll(async () => (await detail()).state, { timeout: 90000 }).toBe('cancelled')
  await page.evaluate(() => window.scrollTo(0, 0))
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
  await page.screenshot({ path: resolve(evidence, 'meter-lab-cancelled.png'), fullPage: true })
  console.log('Verified Start, Pause, navigation/reconnect, Resume, and orderly Stop')
})

test('PV charts use ARM diagnostics, preserve nulls, and synchronize zoom', async ({ page, request }) => {
  test.setTimeout(600000)
  const runs = await (await request.get('/api/v1/meter-lab/runs')).json()
  for (const label of ['pv_warm', 'pv_withheld']) {
    const saved = runs.find(
      (r: any) =>
        r.name === 'Verification: ' + label && r.state === 'completed' && r.checks.verdict === 'pass',
    )
    expect(saved).toBeTruthy()
    const series = await (await request.get('/api/v1/meter-lab/runs/' + saved.id + '/series')).json()
    expect(series.source).toBe('ARM analysis diagnostics')
    await page.goto('/meter-lab?run=' + saved.id)
    await expect(page.getByRole('heading', { name: 'Estimated PV generation', exact: true })).toBeVisible()
    const generation = () =>
      page.evaluate(
        () =>
          Array.from(document.querySelectorAll('.js-plotly-plot'))
            .flatMap((node) => (node as any).data || [])
            .find((trace) => trace.name === 'Estimated PV kW')?.y,
      )
    await expect
      .poll(async () => (await generation())?.length, { timeout: 180000 })
      .toBe(series.diagnostics.length)
    const actual = await generation()
    expect(actual).toEqual(series.diagnostics.map((row: any) => row.generation_kw ?? null))
    if (label === 'pv_warm')
      expect(actual.some((value: number | null) => value != null && value > 0)).toBe(true)
    else expect(actual.every((value: number | null) => value === null)).toBe(true)
    if (label === 'pv_warm') {
      const inputPlot = page
        .getByRole('img', { name: 'Scenario input power/current', exact: true })
        .locator('.js-plotly-plot')
      const outputPlot = page
        .getByRole('img', { name: 'ARM analysis output', exact: true })
        .locator('.js-plotly-plot')
      const range = (plot: typeof inputPlot) =>
        plot.evaluate((node) => (node as any)._fullLayout.xaxis.range as number[])
      const original = await range(inputPlot)
      const drag = inputPlot.locator('.nsewdrag').first()
      await drag.scrollIntoViewIfNeeded()
      const box = await drag.boundingBox()
      expect(box).toBeTruthy()
      await page.mouse.move(box!.x + box!.width * 0.2, box!.y + box!.height * 0.3)
      await page.mouse.down()
      await page.mouse.move(box!.x + box!.width * 0.7, box!.y + box!.height * 0.7, { steps: 8 })
      await page.mouse.up()
      await expect
        .poll(async () => {
          const r = await range(inputPlot)
          return r[1] - r[0]
        })
        .toBeLessThan((original[1] - original[0]) * 0.8)
      await expect
        .poll(async () => {
          const [a, b] = await Promise.all([range(inputPlot), range(outputPlot)])
          return Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]))
        })
        .toBeLessThan(1e-6)
      await page.getByRole('button', { name: 'Reset zoom', exact: true }).click()
      await expect.poll(async () => (await generation())?.length).toBe(series.diagnostics.length)
      await expect
        .poll(async () => {
          const r = await range(inputPlot)
          return r[1] - r[0]
        })
        .toBeGreaterThan((original[1] - original[0]) * 0.95)
      console.log('Verified synchronized scenario-time zoom and reset')
    }
    await page.evaluate(() => window.scrollTo(0, 0))
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
    await page.screenshot({ path: resolve(evidence, 'meter-lab-' + label + '.png'), fullPage: true })
    console.log('Verified actual ARM PV chart: ' + label)
  }
})
