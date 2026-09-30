import { test, expect } from '@playwright/test'

test('scope follows input progress, holds on pause and stop, and uses separate fault scales', async ({
  page,
}) => {
  test.setTimeout(300000)
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  let state = 'running'
  let time = 100
  let agent = 'pv'
  const windows: [number, number][] = []
  const run = () => ({
    id: 'scope-test',
    name: 'Scope test',
    agent,
    state,
    created_at: '2026-09-23T12:00:00Z',
    manifest: { mode: 'replay', meter_form: 'GENX_PP', total_samples: 1000 },
    telemetry: {
      state,
      received: 100,
      processed: agent === 'fault' ? 0 : time,
      scenario_time: time,
      total_samples: 1000,
    },
    checks: { verdict: 'pending', checks: [] },
    checkpoint_available: false,
  })
  await page.route('**/api/v1/meter-lab/**', async (route) => {
    const url = new URL(route.request().url())
    const path = url.pathname.split('/meter-lab')[1]
    let body: unknown = {}
    if (path === '/capabilities') body = { available: true, agents: [], worker_alive: true }
    else if (path === '/scenarios') body = { catalog: [], saved: [] }
    else if (path === '/runs') body = [run()]
    else if (path.endsWith('/series')) {
      const end = Number(url.searchParams.get('end') ?? time)
      const start = Number(url.searchParams.get('start') ?? 0)
      if (url.searchParams.has('end')) windows.push([start, end])
      body = {
        inputs: {
          channels:
            agent === 'pv' ? ['Signed aggregate kW', 'Aggregate kvar'] : ['IA', 'IB', 'IC', 'UA', 'UB', 'UC'],
          rows:
            agent === 'pv'
              ? [
                  [start, 2, 0.3],
                  [end - 1, null, 0.4],
                  [end, -1, 0.5],
                ]
              : [
                  [start, 1, 2, 3, 110, 120, 130],
                  [end, 2, 3, 4, 120, 130, 140],
                ],
        },
        diagnostics: [],
        progress_time: time,
        output_available: 0,
      }
    } else if (path.endsWith('/outcomes')) body = { rows: [], total: 0 }
    else if (path.endsWith('/logs')) body = []
    else if (path.endsWith('/control')) {
      const action = route.request().postDataJSON().action
      state = action === 'pause' ? 'paused' : action === 'stop' ? 'cancelled' : 'running'
      body = run()
    } else body = run()
    await route.fulfill({ json: body })
  })
  await page.goto('/meter-lab?run=scope-test')
  const scope = page.getByRole('region', { name: 'Incoming data scope' })
  const plot = scope.locator('.js-plotly-plot')
  const snapshot = () =>
    plot.evaluate((node: any) => ({
      x: node.data[0].x,
      y: node.data[0].y,
      range: node.layout.xaxis.range,
      axes: node.data.map((trace: any) => trace.yaxis),
      connectgaps: node.data[0].connectgaps,
    }))
  await expect(plot).toBeVisible({ timeout: 180000 })
  await expect.poll(async () => (await snapshot()).range).toEqual([40, 100])
  expect((await snapshot()).y).toEqual([2, null, -1])
  expect((await snapshot()).connectgaps).toBe(false)
  const controls = await page.locator('.ml-run-actions').boundingBox()
  const scopeBox = await scope.boundingBox()
  expect(scopeBox!.y).toBeGreaterThanOrEqual(controls!.y + controls!.height)
  time = 110
  await expect.poll(async () => (await snapshot()).range).toEqual([50, 110])
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  await expect(scope.getByText('Paused', { exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'logs', exact: true }).click()
  await expect(plot).toBeVisible()
  expect((await snapshot()).range).toEqual([50, 110])
  await page.getByRole('button', { name: 'Resume', exact: true }).click()
  time = 120
  await expect.poll(async () => (await snapshot()).range).toEqual([60, 120])
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect(scope.getByText('cancelled', { exact: true })).toBeVisible()
  expect((await snapshot()).range).toEqual([60, 120])
  expect(windows.every(([start, end]) => end - start === 60)).toBe(true)
  agent = 'fault'
  state = 'running'
  time = 0.1
  await page.reload()
  await expect.poll(async () => (await snapshot()).axes).toEqual(['y', 'y', 'y', 'y2', 'y2', 'y2'])
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
    .toBe(true)
  await scope.screenshot({ path: `../runtime/meter-lab/verification/scope-${test.info().project.name}.png` })
  if (test.info().project.name === 'desktop') {
    for (const [name, width] of [['tablet', 820], ['phone', 390]] as const) {
      await page.setViewportSize({ width, height: 1000 })
      await expect(plot).toBeVisible()
      await expect
        .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
        .toBe(true)
      await scope.screenshot({ path: `../runtime/meter-lab/verification/scope-${name}.png` })
    }
  }
  expect(errors).toEqual([])
})
