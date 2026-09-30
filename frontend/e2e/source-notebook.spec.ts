import {test,expect} from '@playwright/test'

test('power anomaly source is available without inferred clock or channel metadata',async({page,context})=>{
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
  await page.getByLabel('Notebook data source').selectOption('source:power_anomaly_dataset_2022.csv')
  await expect(page.getByText('Source table',{exact:true})).toBeVisible()
  await page.getByRole('button',{name:'Data guide',exact:true}).click()
  await expect(page.locator('.notebook-guide .code')).toContainText('from di_sources import read_csv')
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
  await expect.poll(()=>page.evaluate(()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement)?.contentWindow as any
    return w?.jupyterapp?.shell.currentWidget?.sessionContext?.kernelDisplayStatus
  }),{timeout:90000}).toBe('idle')
  await page.evaluate(async()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    const panel=await w.jupyterapp.commands.execute('docmanager:open',{path:'03 - Power anomaly exploration.ipynb'})
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
    const cells=w.jupyterapp.shell.currentWidget.model.toJSON().cells.filter((c:any)=>c.cell_type==='code')
    return cells.every((c:any)=>c.execution_count!==null)
  }),{timeout:90000}).toBe(true)
  const result=await page.evaluate(()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    const outputs=w.jupyterapp.shell.currentWidget.model.toJSON().cells.flatMap((c:any)=>c.outputs||[])
    return {errors:outputs.filter((o:any)=>o.output_type==='error'),text:JSON.stringify(outputs),images:outputs.filter((o:any)=>o.data?.['image/png']).length}
  })
  expect(result.errors).toEqual([])
  expect(result.text).toContain('(1300, 9)')
  expect(result.text).toMatch(/'source_checksum': '[a-f0-9]{64}'/)
  expect(result.text).toContain('Cyber_Spike')
  expect(result.text).toContain('timezone: None')
  expect(result.text).toContain('Exploratory flagged samples:')
  expect(result.images).toBeGreaterThan(0)
  expect(external).toEqual([])
})
