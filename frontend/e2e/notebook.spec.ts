import {test,expect} from '@playwright/test'

test('local notebook executes Python against AMI and recording data and retains its session',async({page,context})=>{
  test.setTimeout(240000)
  const external:string[]=[]
  await context.route('**/*',route=>{
    const url=new URL(route.request().url())
    if(!['127.0.0.1','localhost'].includes(url.hostname)&&url.protocol.startsWith('http')){
      external.push(url.href)
      return route.abort()
    }
    return route.continue()
  })
  await page.goto('/notebook')
  await expect(page.getByRole('heading',{name:'Notebook',exact:true})).toBeVisible()
  await expect(page.getByRole('link',{name:'Open in new tab',exact:true})).toHaveAttribute('href',/\/notebooks\/lab\//)
  await page.getByRole('button',{name:'Data guide',exact:true}).click()
  await expect(page.locator('.notebook-guide .code')).toContainText('await di.readings')
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
  const iframe=page.locator('iframe[title="JupyterLite notebook workspace"]')
  await expect(iframe).toBeVisible()
  await expect.poll(()=>page.evaluate(()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement)?.contentWindow as any
    return w?.jupyterapp?.shell.currentWidget?.sessionContext?.kernelDisplayStatus
  }),{timeout:120000}).toBe('idle')
  await expect.poll(()=>page.evaluate(()=>{
    const element=document.querySelector('iframe') as HTMLIFrameElement
    const shell=(element.contentWindow as any).jupyterapp.shell
    return element.clientWidth>=650||shell.leftCollapsed
  })).toBe(true)
  const code=`import pandas as pd
import matplotlib.pyplot as plt
from di_data import DIClient
di = DIClient()
datasets = await di.datasets()
dataset = next(d for d in datasets if d['format'] == 'wide_ami')
assets = await di.assets(dataset['id'])
page = await di.readings(dataset['id'], asset_ids=[assets[0]['id']], limit=17)
frame = di.to_frame(page)
assert len(frame) == 17
assert frame.attrs['resolution'] == 'native'
assert pd.api.types.is_numeric_dtype(frame['value'])
recording = next(d for d in datasets if d['format'] == 'recording' and d['sample_rate'] == 1000 and d['duration_seconds'] >= 10.5 and any(c['name'] == 'power' for c in d['channels']))
samples = di.to_frame(await di.readings(recording['id'], start=9.5, end=10.5, channels=['power']))
assert len(samples) == 1001
assert samples.attrs['synthetic']
assert abs(samples['offset'].iloc[0] - 9.5) < 1e-9
samples.plot(x='offset', y='power')
plt.show()
notebook_verification = 'DI_NOTEBOOK_OK'
print(notebook_verification, len(frame), len(samples))`
  await page.evaluate(code=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    const panel=w.jupyterapp.shell.currentWidget
    panel.content.activeCellIndex=1
    panel.content.activeCell.model.sharedModel.setSource(code)
    void w.jupyterapp.commands.execute('notebook:run-cell')
  },code)
  const notebook=page.frameLocator('iframe[title="JupyterLite notebook workspace"]')
  await expect(notebook.locator('.jp-OutputArea-output').filter({hasText:'DI_NOTEBOOK_OK 17 1001'})).toBeVisible({timeout:120000})
  await expect(notebook.locator('.jp-OutputArea-output img').first()).toBeVisible()
  await expect(notebook.locator('.jp-OutputArea-error')).toHaveCount(0)
  await page.evaluate(async()=>{
    const w=(document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any
    await w.jupyterapp.shell.currentWidget.context.save()
  })
  // Use app navigation (not full reload) to verify that the iframe/kernel persists.
  async function navigate(name:string){
    if(await page.getByRole('button',{name:'Toggle navigation',exact:true}).isVisible())
      await page.getByRole('button',{name:'Toggle navigation',exact:true}).click()
    await page.getByRole('navigation',{name:'Main navigation'}).getByRole('link',{name,exact:true}).click()
  }
  const kernel=await page.evaluate(()=>((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp.shell.currentWidget.sessionContext.session.kernel.id)
  await navigate('Datasets')
  await expect(page.getByRole('heading',{name:'Dataset workspace',exact:true})).toBeVisible()
  await navigate('Notebook')
  await expect(iframe).toBeVisible()
  expect(await page.evaluate(()=>((document.querySelector('iframe') as HTMLIFrameElement).contentWindow as any).jupyterapp.shell.currentWidget.sessionContext.session.kernel.id)).toBe(kernel)
  await page.reload()
  await expect(notebook.locator('.jp-OutputArea-output').filter({hasText:'DI_NOTEBOOK_OK 17 1001'})).toBeVisible({timeout:90000})
  expect(external).toEqual([])
})
