import {test,expect} from '@playwright/test'

test('COMTRADE catalog, fault detection, distance and exports',async({page,request})=>{
  test.setTimeout(180000)
  const errors:string[]=[]
  page.on('pageerror',e=>errors.push(e.message))
  const sources=await (await request.get('/api/v1/faults/sources')).json()
  expect(sources).toHaveLength(99)
  const source=sources.find((s:any)=>s.name==='1A_val1')
  await page.goto('/events')
  // Import order used to select val99, while the notebook computes val1.
  await expect(page.getByRole('combobox',{name:'Recording',exact:true})).toHaveValue(source.dataset_id)
  await page.getByRole('button',{name:'COMTRADE recordings',exact:true}).click()
  await page.getByRole('checkbox',{name:'Import 1A_val1',exact:true}).check()
  const queued=page.waitForResponse(r=>r.url().endsWith('/api/v1/faults/imports')&&r.request().method()==='POST')
  await page.getByRole('button',{name:'Import 1 selected',exact:true}).click()
  const imported=await (await queued).json()
  await expect.poll(async()=> (await (await request.get('/api/v1/jobs/'+imported.id)).json()).status,{timeout:90000}).toBe('completed')
  await expect(page.getByText('Imported recordings are ready to open above.')).toBeVisible()
  await page.getByRole('row').filter({has:page.getByRole('checkbox',{name:'Import 1A_val1',exact:true})}).getByRole('button',{name:'Open in Event Lab'}).click()
  await expect(page.getByRole('combobox',{name:'Detector adapter',exact:true})).toHaveValue('fault-distance')
  await expect(page.getByText('Offline waveform analysis - native samples',{exact:true})).toBeVisible()
  await expect(page.getByText(/timezone unverified/)).toBeVisible()
  await expect(page.getByRole('checkbox',{name:'Line parameters verified for this recording',exact:true})).not.toBeChecked()
  await page.getByLabel('X1 (Ω/km)',{exact:true}).fill('0.4')
  const submit=page.waitForResponse(r=>r.url().endsWith('/api/v1/events/run')&&r.request().method()==='POST')
  await page.getByRole('button',{name:'Run detectors',exact:true}).click()
  const response=await submit
  expect(response.ok()).toBe(true)
  const job=await response.json()
  expect(job.config.algorithm).toBe('fault-distance')
  expect(job.config.parameters.x1_ohm_km).toBe(.4)
  await expect.poll(async()=> (await (await request.get('/api/v1/jobs/'+job.id)).json()).status,{timeout:90000}).toBe('completed')
  const finished=await (await request.get('/api/v1/jobs/'+job.id)).json()
  await expect(page.getByRole('combobox',{name:'Detection run',exact:true})).toHaveValue(finished.result.id)
  await expect(page.getByText('Detection completed. Results are ready.',{exact:true})).toBeVisible()
  const result=await (await request.get('/api/v1/experiments/'+finished.result.id)).json()
  expect(result.fault_analyses[0].distance.estimated_distance_km).toBeCloseTo(.89089036,6)
  expect(result.events).toHaveLength(1)
  expect(result.events[0].kind).toBe('fault')
  await expect(page.getByRole('heading',{name:'Fault distance analysis',exact:true})).toBeVisible()
  await expect(page.locator('.fault-distance-value')).toContainText('0.8909')
  await expect(page.locator('.fault-source')).toHaveText('Recording: 1A_val1')
  await expect(page.getByText('Uncompensated apparent distance',{exact:true})).toBeVisible()
  await expect(page.getByRole('button',{name:'Fault inception',exact:true})).toBeVisible()
  await page.getByRole('button',{name:'Fault inception',exact:true}).click()
  const exported=await request.get('/api/v1/experiments/'+finished.result.id+'/export')
  expect(exported.ok()).toBe(true)
  expect((await exported.body()).subarray(0,2).toString()).toBe('PK')
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
  await page.screenshot({path:`../runtime/verification/fault-events-${test.info().project.name}.png`,fullPage:true})
  const dataset=await (await request.get('/api/v1/datasets/'+result.dataset_ids[0])).json()
  await page.goto('/')
  await page.getByLabel('Search datasets',{exact:true}).fill(dataset.name)
  await page.getByRole('button',{name:dataset.name,exact:true}).click()
  await expect(page.getByText('Elapsed time; absolute clock unverified',{exact:true})).toBeVisible()
  await expect(page.getByText(/UTC epoch is an elapsed-time storage reference/)).toBeVisible()
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
  await page.getByRole('button',{name:'Close dialog',exact:true}).click()
  expect(errors).toEqual([])
  expect(source.id).toBeTruthy()
})

test('fault recording selection matches the notebook and respects explicit navigation',async({page,request})=>{
  const sources=await (await request.get('/api/v1/faults/sources')).json()
  const first=sources.find((s:any)=>s.name==='1A_val1')
  const last=sources.find((s:any)=>s.name==='1A_val99')
  await page.goto('/events?dataset='+last.dataset_id)
  const recording=page.getByRole('combobox',{name:'Recording',exact:true})
  await expect(recording).toHaveValue(last.dataset_id)
  await expect(page.locator('.fault-source')).toHaveText('Recording: 1A_val99')
  await expect(page.locator('.fault-distance-value')).toContainText('87.1208')
  await page.getByLabel('X1 (Ω/km)',{exact:true}).fill('0.8')
  await page.getByRole('button',{name:'Use notebook example',exact:true}).click()
  await expect(recording).toHaveValue(first.dataset_id)
  await expect(page.getByLabel('X1 (Ω/km)',{exact:true})).toHaveValue('0.4')
  await expect(page.locator('.fault-source')).toHaveText('Recording: 1A_val1')
  await expect(page.locator('.fault-distance-value')).toContainText('0.8909')
  await page.goBack()
  await expect(recording).toHaveValue(last.dataset_id)
  await expect(page.locator('.fault-source')).toHaveText('Recording: 1A_val99')
  await page.goForward()
  await expect(recording).toHaveValue(first.dataset_id)
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
})

test('fault_distance notebook runs the imported pipeline locally',async({page,context,request})=>{
  test.setTimeout(300000)
  const external:string[]=[]
  await context.route('**/*',route=>{
    const url=new URL(route.request().url())
    if(url.protocol.startsWith('http')&&!['127.0.0.1','localhost'].includes(url.hostname)){
      external.push(url.href);return route.abort()
    }
    return route.continue()
  })
  await page.goto('/notebook')
  const iframe=page.frameLocator('iframe[title="JupyterLite notebook workspace"]')
  await expect.poll(()=>page.evaluate(()=>{
    const app=((document.querySelector('iframe') as HTMLIFrameElement)?.contentWindow as any)?.jupyterapp
    return app?.shell.currentWidget?.sessionContext?.kernelDisplayStatus
  }),{timeout:120000}).toBe('idle')
  await page.getByRole('combobox',{name:'Open notebook',exact:true}).selectOption('fault_distance.ipynb')
  const sources=await (await request.get('/api/v1/faults/sources')).json()
  const source=sources.find((s:any)=>s.name==='1A_val1')
  await expect(page.getByRole('combobox',{name:'Notebook data source',exact:true})).toHaveValue(source.dataset_id)
  await expect(page.getByRole('link',{name:'Open 1A_val1 in Event Lab',exact:true})).toHaveAttribute('href','/events?dataset='+source.dataset_id)
  await expect.poll(()=>page.evaluate(()=>{
    const app=((document.querySelector('iframe') as HTMLIFrameElement)?.contentWindow as any)?.jupyterapp
    return app?.shell.currentWidget?.context?.path
  })).toBe('fault_distance.ipynb')
  await expect.poll(()=>page.evaluate(()=>{
    const app=((document.querySelector('iframe') as HTMLIFrameElement)?.contentWindow as any)?.jupyterapp
    return app?.shell.currentWidget?.sessionContext?.kernelDisplayStatus
  }),{timeout:120000}).toBe('idle')
  await page.evaluate(()=>{
    const app=((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp
    void app.commands.execute('notebook:run-all-cells')
  })
  // Inspect model outputs as well as DOM: JupyterLab virtualizes long notebooks.
  await expect.poll(()=>page.evaluate(()=>{
    const app=((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp
    const cells=app.shell.currentWidget.model.toJSON().cells
    const errors=cells.flatMap((c:any)=>c.outputs||[]).filter((o:any)=>o.output_type==='error')
    if(errors.length)return JSON.stringify(errors)
    return cells[39].execution_count?'finished':'running'
  }),{timeout:180000}).toBe('finished')
  await page.evaluate(()=>{
    const panel=((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp.shell.currentWidget
    panel.content.activeCellIndex=39
    void panel.content.scrollToItem(39)
  })
  await expect(iframe.locator('.jp-OutputArea-output').filter({hasText:'Uncompensated apparent distance: 0.8909 km'})).toBeVisible({timeout:180000})
  await expect.poll(()=>page.evaluate(()=>{
    const app=((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp
    return app.shell.currentWidget.sessionContext.kernelDisplayStatus
  }),{timeout:120000}).toBe('idle')
  await expect(iframe.locator('.jp-OutputArea-error')).toHaveCount(0)
  const state=await page.evaluate(async()=>{
    const app=((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp
    const panel=app.shell.currentWidget
    const result=await panel.sessionContext.session.kernel.requestExecute({code:"assert X.shape == (12, 400)\nassert abs(estimated_distance_km - 0.89089036) < 1e-6\nassert source_metadata['clock_verified'] is False\nassert (OUTPUT_FOLDER / 'fault_sample_processed.npz').exists()\nassert (OUTPUT_FOLDER / 'fault_distance_estimate.json').exists()"}).done
    await panel.context.save()
    return result.content.status
  })
  expect(state).toBe('ok')
  expect(external).toEqual([])
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
  await page.screenshot({path:`../runtime/verification/fault-notebook-${test.info().project.name}.png`,fullPage:true})
})
