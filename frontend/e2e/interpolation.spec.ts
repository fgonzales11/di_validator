import {test,expect} from '@playwright/test'

test('interpolation checkbox repairs eligible gaps with an auditable worker result',async({page,request})=>{
  test.setTimeout(150000)
  const name='Interpolation fixture '+test.info().project.name+' '+Date.now()
  const missing=new Set([160,161,225,232])
  const csv='Timestamp,power\n'+Array.from({length:240},(_,i)=>
    `${new Date(Date.UTC(2025,0,1)+i*3600000).toISOString()},${missing.has(i)?'':(-8+i*.05).toFixed(5)}`
  ).join('\n')
  const upload=await request.post('/api/v1/uploads',{multipart:{file:{name:'interpolation-fixture.csv',mimeType:'text/csv',buffer:Buffer.from(csv)}}})
  expect(upload.ok()).toBe(true)
  const queued=await request.post('/api/v1/datasets',{data:{path:(await upload.json()).path,name,format:'recording',asset_id:name,
    timestamp_column:'Timestamp',timezone:'UTC',sample_rate:1/3600,channels:[{name:'power',kind:'power',unit:'kW'}]}})
  expect(queued.ok()).toBe(true)
  const imported=await queued.json()
  await expect.poll(async()=> (await (await request.get('/api/v1/jobs/'+imported.id)).json()).status,{timeout:60000}).toBe('completed')
  const importJob=await (await request.get('/api/v1/jobs/'+imported.id)).json()
  const errors:string[]=[]
  page.on('pageerror',e=>errors.push(e.message))
  await page.goto('/forecasting')
  await page.getByLabel('Forecast data source').selectOption(importJob.result.id)
  const checkbox=page.getByRole('checkbox',{name:'Interpolate missing data',exact:true})
  await expect(checkbox).not.toBeChecked()
  await checkbox.check()
  await page.getByLabel('Maximum interpolated gap (steps)',{exact:false}).fill('2')
  await checkbox.uncheck()
  await expect(page.getByRole('combobox',{name:'Missing data',exact:true})).toHaveValue('reject')
  await checkbox.check()
  await page.reload()
  await expect(checkbox).toBeChecked()
  await expect(page.getByLabel('Maximum interpolated gap (steps)',{exact:false})).toHaveValue('2')
  for(const [label,value] of [['Forecast name',name],['Forecast steps','12'],['Validation windows','2'],['Training context (steps)','96'],['Maximum history (steps)','300'],['Lag features','12']]){
    await page.getByLabel(label,{exact:true}).fill(value)
  }
  for(const model of ['Random Forest','XGBoost','LightGBM','Ensemble · equal-weight mean']){
    await page.getByRole('checkbox',{name:model,exact:true}).uncheck()
  }
  await page.getByRole('button',{name:'Save preset',exact:true}).click()
  const submit=page.waitForResponse(r=>r.url().endsWith('/api/v1/forecasting/runs')&&r.request().method()==='POST')
  await page.getByRole('button',{name:'Run forecast benchmark',exact:true}).click()
  const response=await submit
  expect(response.ok()).toBe(true)
  const job=await response.json()
  expect(job.config.missing_policy).toBe('interpolate')
  expect(job.config.max_gap_steps).toBe(2)
  await page.getByRole('button',{name:'Open completed forecast',exact:true}).click({timeout:60000})
  const audit=page.getByLabel('Interpolation audit',{exact:true})
  await expect(audit).toContainText('Future forecast training: 4 values in 3 gaps interpolated')
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
  const done=await (await request.get('/api/v1/jobs/'+job.id)).json()
  const run=await (await request.get('/api/v1/forecasting/runs/'+done.result.id)).json()
  expect(run.source.missing).toBe(4)
  expect(run.models[0].holdout.n).toBe(11)
  expect(run.preprocessing.map((p:any)=>p.filled)).toEqual([2,2,3,4])
  await page.screenshot({path:`../runtime/verification/interpolation-${test.info().project.name}.png`,fullPage:true})
  await page.getByRole('button',{name:'Close dialog'}).click()
  await checkbox.uncheck()
  const presets=await (await request.get('/api/v1/presets')).json()
  await page.getByLabel('Forecast preset',{exact:true}).selectOption(presets.find((p:any)=>p.name===name).id)
  await expect(checkbox).toBeChecked()
  await expect(page.getByLabel('Maximum interpolated gap (steps)',{exact:false})).toHaveValue('2')
  expect(errors).toEqual([])
})
