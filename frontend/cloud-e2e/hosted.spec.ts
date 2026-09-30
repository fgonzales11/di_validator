import {test, expect, type APIRequestContext} from '@playwright/test'

async function finished(request: APIRequestContext, id: string) {
  let job: any
  await expect.poll(async () => {
    const response = await request.get('/api/v1/jobs/' + id)
    expect(response.ok()).toBe(true)
    job = await response.json()
    if (['failed', 'cancelled', 'interrupted'].includes(job.status)) throw new Error(JSON.stringify(job))
    return job.status
  }, {timeout: 240000, intervals: [1000, 2000, 4000]}).toBe('completed')
  return job.result
}

test('hosted data, fault calculation, forecast training, saved inference and exports', async ({request}) => {
  const catalog = await request.get('/api/v1/faults/sources')
  expect(catalog.ok()).toBe(true)
  const sources = await catalog.json()
  expect(sources).toHaveLength(99)
  expect(sources.every((item: any) => item.dataset_id)).toBe(true)
  const example = sources.find((item: any) => item.name === '1A_val1')
  const datasets = await (await request.get('/api/v1/datasets')).json()
  expect(datasets.filter((item: any) => item.format === 'wide_ami').length).toBeGreaterThanOrEqual(6)
  const native = await (await request.get(`/api/v1/notebooks/data/${example.dataset_id}?start=0&end=0.1995`)).json()
  expect(native.rows).toHaveLength(400)
  const queued = await request.post('/api/v1/events/run', {data: {dataset_id: example.dataset_id, algorithm: 'fault-distance'}})
  expect(queued.ok()).toBe(true)
  const faultJob = await finished(request, (await queued.json()).id)
  const fault = await (await request.get('/api/v1/experiments/' + faultJob.id)).json()
  expect(fault.fault_analyses[0].distance.estimated_distance_km).toBeCloseTo(.89089036, 6)
  const artifact = await request.get(`/api/v1/experiments/${fault.id}/export`)
  expect(artifact.ok()).toBe(true)
  expect((await artifact.body()).subarray(0, 2).toString()).toBe('PK')

  const models = await (await request.get('/api/v1/forecasting/models')).json()
  expect(models.every((model: any) => model.ready)).toBe(true)
  expect(models.some((model: any) => ['chronos', 'timesfm', 'tabpfn'].includes(model.id))).toBe(false)
  const config = {name: 'Cloud deployment verification', source_file: 'power_anomaly_dataset_2022.csv', channel: 'Load_kW',
    unit: 'kW', horizon: 12, validation_windows: 2, context_length: 128, max_history: 300,
    models: ['seasonal_naive', 'random_forest', 'xgboost', 'lightgbm', 'ensemble'],
    parameters: {random_forest: {n_estimators: 20}, xgboost: {n_estimators: 20}, lightgbm: {n_estimators: 20}}}
  const submitted = await request.post('/api/v1/forecasting/runs', {data: config})
  expect(submitted.ok()).toBe(true)
  const forecast = await finished(request, (await submitted.json()).id)
  const predictions = await request.get(`/api/v1/forecasting/runs/${forecast.id}/predictions`)
  expect(predictions.ok()).toBe(true)
  expect((await predictions.json()).length).toBeGreaterThan(12)
  const exportResult = await request.get(`/api/v1/forecasting/runs/${forecast.id}/export`)
  expect(exportResult.ok()).toBe(true)
  expect((await exportResult.body()).subarray(0, 2).toString()).toBe('PK')
  const reused = await request.post('/api/v1/forecasting/runs', {data: {...config, models: ['random_forest'],
    parameters: {random_forest: config.parameters.random_forest},
    reuse_run_id: forecast.id, reuse_model: 'random_forest'}})
  expect(reused.ok()).toBe(true)
  const inferenceJob = await finished(request, (await reused.json()).id)
  const inference = await (await request.get('/api/v1/forecasting/runs/' + inferenceJob.id)).json()
  expect(inference.mode).toBe('inference')
})

test('hosted workspace navigation, charts and responsive layouts', async ({page}) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  for (const width of [1440, 820, 390]) {
    await page.setViewportSize({width, height: 1000})
    for (const [route, heading] of [['/', 'Dataset workspace'], ['/forecasting', 'Forecasting workbench'], ['/events', 'Event lab']]) {
      await page.goto(route)
      await expect(page.getByRole('heading', {name: heading, exact: true})).toBeVisible()
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
    }
    await expect(page.locator('.fault-source')).toHaveText('Recording: 1A_val1')
    await expect(page.locator('.fault-distance-value')).toContainText('0.8909')
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible()
    await expect.poll(() => page.locator('.js-plotly-plot').first().evaluate((element: any) =>
      element.data?.some((trace: any) => trace.y?.length >= 400 && Array.from(trace.y).some(value => Number.isFinite(value))))).toBe(true)
    await page.screenshot({path: `../runtime/cloud-deploy/hosted-${width}.png`, fullPage: true})
  }
  expect(errors).toEqual([])
})

test('hosted JupyterLite runs Python and reads native cloud data', async ({page}) => {
  page.on('response', response => {
    if (response.status() >= 400) console.log('Notebook HTTP error', response.status(), new URL(response.url()).pathname)
  })
  await page.goto('/notebook')
  await expect(page.getByRole('heading', {name: 'Notebook', exact: true})).toBeVisible()
  expect(await page.evaluate(() => window.crossOriginIsolated)).toBe(true)
  await expect.poll(() => page.evaluate(() => {
    const app = ((document.querySelector('iframe') as HTMLIFrameElement)?.contentWindow as any)?.jupyterapp
    return app?.shell.currentWidget?.sessionContext?.kernelDisplayStatus
  }), {timeout: 180000}).toBe('idle')
  const code = `import pandas as pd
from di_data import DIClient
di = DIClient()
datasets = await di.datasets()
ami = next(d for d in datasets if d['format'] == 'wide_ami')
assets = await di.assets(ami['id'])
rows = await di.readings(ami['id'], asset_ids=[assets[0]['id']], limit=17)
frame = di.to_frame(rows)
assert len(frame) == 17
assert frame.attrs['resolution'] == 'native'
recording = next(d for d in datasets if d['name'].endswith(' 1A_val1'))
native = await di.readings(recording['id'], start=0, end=0.1995)
assert len(native['rows']) == 400
print('DI_CLOUD_NOTEBOOK_OK', len(frame), len(native['rows']))`
  await page.evaluate(code => {
    const app = ((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp
    const panel = app.shell.currentWidget
    panel.content.activeCellIndex = 1
    panel.content.activeCell.model.sharedModel.setSource(code)
    void app.commands.execute('notebook:run-cell')
  }, code)
  await expect.poll(() => page.evaluate(() => {
    const app = ((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp
    return app.shell.currentWidget.model.toJSON().cells[1].execution_count
  }), {timeout: 120000}).not.toBeNull()
  await expect.poll(() => page.evaluate(() => {
    const app = ((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp
    return app.shell.currentWidget.sessionContext.kernelDisplayStatus
  }), {timeout: 120000}).toBe('idle')
  const outputs = await page.evaluate(() => {
    const app = ((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp
    return app.shell.currentWidget.model.toJSON().cells[1].outputs
  })
  expect(outputs.filter((output: any) => output.output_type === 'error')).toEqual([])
  expect(JSON.stringify(outputs)).toContain('DI_CLOUD_NOTEBOOK_OK 17 400')
  const notebook = page.frameLocator('iframe[title="JupyterLite notebook workspace"]')
  await expect(notebook.locator('.jp-OutputArea-output').filter({hasText: 'DI_CLOUD_NOTEBOOK_OK 17 400'})).toBeVisible({timeout: 120000})
  await expect(notebook.locator('.jp-OutputArea-error')).toHaveCount(0)
  await page.screenshot({path: '../runtime/cloud-deploy/hosted-notebook.png', fullPage: true})
})
