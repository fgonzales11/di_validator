"""Run local forecasting jobs from browser notebooks (ML runs in the worker)."""
import json
from urllib.parse import quote


async def _request(path, body=None):
    from js import location
    from pyodide.http import pyfetch
    options = {} if body is None else dict(method="POST",headers={"Content-Type":"application/json"},body=json.dumps(body))
    response = await pyfetch(str(location.origin)+"/api/v1"+path,**options)
    result = await response.json()
    if not response.ok:
        raise ValueError(result.get("detail",str(response.status)))
    return result


async def models():
    return await _request("/forecasting/models")


async def start(config):
    return await _request("/forecasting/runs",config)


async def job(job_id):
    return await _request("/jobs/"+quote(job_id,safe=""))


async def cancel(job_id):
    return await _request("/jobs/"+quote(job_id,safe="")+"/cancel",{})


async def runs():
    return await _request("/forecasting/runs")


async def predictions(run_id, model):
    import pandas as pd
    return pd.DataFrame(await _request("/forecasting/runs/"+quote(run_id,safe="")+"/predictions?model="+quote(model,safe="")))
