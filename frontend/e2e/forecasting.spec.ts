import {test,expect} from '@playwright/test'

test('forecast source benchmark, export, saved inference, and model restrictions',async({page,request})=>{
  const errors:string[]=[]
  page.on('pageerror',error=>errors.push(error.message))
  await page.goto('/forecasting')
  await expect(page.getByRole('heading',{name:'Forecasting workbench',exact:true})).toBeVisible()
  await page.getByLabel('Forecast data source').selectOption('source:power_anomaly_dataset_2022.csv')
  await expect(page.getByLabel('Forecast measurement')).toHaveValue('Load_kW')
  await expect(page.getByLabel('Forecast measurement').locator('option')).not.toContainText(['Event_Label'])
  await page.getByLabel('Forecast name',{exact:true}).fill('Browser forecast '+test.info().project.name)
  await page.getByLabel('Forecast steps',{exact:true}).fill('12')
  await page.getByLabel('Validation windows',{exact:true}).fill('2')
  await page.getByLabel('Training context (steps)').fill('128')
  await page.getByLabel('Maximum history (steps)').fill('300')
  await page.getByLabel('XGBoost',{exact:true}).uncheck()
  await page.getByLabel('LightGBM',{exact:true}).uncheck()
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
  await page.getByRole('button',{name:'Run forecast benchmark',exact:true}).click()
  await page.getByRole('button',{name:'Open completed forecast',exact:true}).click({timeout:90000})
  const dialog=page.getByRole('dialog')
  await expect(dialog.getByText('Source provenance unverified',{exact:true})).toBeVisible()
  await expect(dialog.locator('.js-plotly-plot')).toBeVisible()
  await expect.poll(()=>dialog.locator('.js-plotly-plot').evaluate((element:any)=>element.data?.find((trace:any)=>trace.name==='Forecast')?.y?.length)).toBe(12)
  const download=await Promise.all([page.waitForEvent('download'),dialog.getByRole('link',{name:'Export forecast bundle'}).click()])
  expect(download[0].suggestedFilename()).toMatch(/^di-forecast-.*\.zip$/)
  await dialog.getByLabel('Forecast result model').selectOption('random_forest')
  await dialog.getByRole('button',{name:'Use saved fitted model'}).click()
  await page.getByRole('button',{name:'Predict with saved model'}).click()
  await page.getByRole('button',{name:'Open completed forecast',exact:true}).click({timeout:90000})
  await expect(dialog.getByText('Inference from a saved fitted model; no training or evaluation on this run.')).toBeVisible()
  await expect(dialog.getByLabel('Forecast view')).toHaveValue('future')
  await dialog.getByRole('button',{name:'Close dialog'}).click()
  await page.getByRole('tab',{name:'Model library'}).click()
  await expect(page.getByRole('heading',{name:'AutoGluon tree ensemble',exact:true})).toBeVisible()
  const models=await (await request.get('/api/v1/forecasting/models')).json()
  expect(models.some((m:any)=>['chronos','timesfm','tabpfn'].includes(m.id))).toBe(false)
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
  expect(errors).toEqual([])
})

test('forecast notebook submits a local worker job and plots results',async({page,context})=>{
  test.skip(test.info().project.name!=='desktop','Kernel integration is exercised once; forecasting layout is checked at all widths.')
  test.setTimeout(180000)
  const external:string[]=[]
  await context.route('**/*',route=>{
    const url=new URL(route.request().url())
    if(url.protocol.startsWith('http')&&!['127.0.0.1','localhost'].includes(url.hostname)){
      external.push(url.href)
      return route.abort()
    }
    return route.continue()
  })
  await page.goto('/notebook')
  await expect.poll(()=>page.evaluate(()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement)?.contentWindow as any
    return w?.jupyterapp?.shell.currentWidget?.sessionContext?.kernelDisplayStatus
  }),{timeout:90000}).toBe('idle')
  await page.evaluate(async()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    const panel=await w.jupyterapp.commands.execute('docmanager:open',{path:'04 - Forecasting.ipynb'})
    await panel.context.ready
    await panel.sessionContext.ready
  })
  await expect.poll(()=>page.evaluate(()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    return w.jupyterapp.shell.currentWidget.sessionContext.kernelDisplayStatus
  }),{timeout:60000}).toBe('idle')
  await page.evaluate(()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    void w.jupyterapp.commands.execute('notebook:run-all-cells')
  })
  await expect.poll(()=>page.evaluate(()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    return w.jupyterapp.shell.currentWidget.model.toJSON().cells.filter((c:any)=>c.cell_type==='code').every((c:any)=>c.execution_count!==null)
  }),{timeout:90000}).toBe(true)
  const output=await page.evaluate(()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    const outputs=w.jupyterapp.shell.currentWidget.model.toJSON().cells.flatMap((c:any)=>c.outputs||[])
    return {errors:outputs.filter((o:any)=>o.output_type==='error'),text:JSON.stringify(outputs),images:outputs.filter((o:any)=>o.data?.['image/png']).length}
  })
  expect(output.errors).toEqual([])
  expect(output.text).toContain('Completed:')
  expect(output.images).toBeGreaterThan(0)
  expect(external).toEqual([])
})
