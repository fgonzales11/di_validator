"""Opt-in real-provider smoke checks; artifacts stay under runtime/verification."""
import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

from di_validator import store
from di_validator.forecasting.catalog import catalog, interpreter
from di_validator.forecasting.contracts import ForecastConfig
from di_validator.forecasting.engine import fitted_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['seasonal_naive','random_forest','xgboost','lightgbm','prophet','gluonts','autogluon','neuralprophet'])
    args = parser.parse_args()
    folder = store.workspace()/'verification'/'forecasting'/store.uid()
    folder.mkdir(parents=True)
    series = pd.Series(5+np.sin(np.arange(320)*2*np.pi/24)+np.arange(320)*.002,
                       index=pd.date_range('2025-01-01',periods=320,freq='h'))
    settings = {'gluonts':{'epochs':1,'num_batches_per_epoch':2,'hidden_size':10},
                'neuralprophet':{'epochs':2,'learning_rate':.01},'autogluon':{'time_limit':20}}
    progress, results = store.Progress(), []
    from di_validator.forecasting.catalog import checked_parameters
    for model in args.models:
        start = time.monotonic()
        print('Checking '+model,flush=True)
        try:
            cfg = ForecastConfig(source_file='fixture.csv',channel='power',horizon=12,lags=12,
                                 context_length=320,max_history=320,season_length=24,models=[model],
                                 parameters={model:checked_parameters(model,settings.get(model,{}))})
            dependencies = subprocess.check_output([str(interpreter(model)),'-c',
                "import importlib.util,json; print(json.dumps({m:importlib.util.find_spec(m) is not None for m in ['huggingface_hub','transformers']}))"],text=True)
            assert not any(json.loads(dependencies).values()),dependencies
            first = fitted_prediction(model,series,cfg,3600000000000,folder/model/'fitted',progress)
            reused = fitted_prediction(model,series,cfg,3600000000000,folder/model/'reused',progress,reuse=folder/model/'fitted')
            np.testing.assert_allclose(first,reused,rtol=1e-6,atol=1e-6)
            results.append(dict(model=model,status='passed',seconds=round(time.monotonic()-start,2),prediction=first.tolist()))
        except Exception as error:
            results.append(dict(model=model,status='failed',error=str(error)))
        print(json.dumps(results[-1]),flush=True)
        (folder/'results.json').write_text(json.dumps(dict(models=catalog(),checks=results),indent=2),encoding='utf-8')
    print('Report: '+str(folder/'results.json'))
    if any(r['status']=='failed' for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
