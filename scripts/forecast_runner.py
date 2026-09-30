"""File protocol for local, isolated forecasting providers. No uploaded Python code."""
from pathlib import Path
import json
import os
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("DO_NOT_TRACK","1")
os.environ.setdefault("OMP_NUM_THREADS","2")


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    response = Path(request["response"])
    try:
        import numpy as np
        from di_validator.forecasting import providers
        if request.get("mode") == "probe":
            import importlib
            modules={"gluonts":"gluonts.torch.model.deepar", "autogluon":"autogluon.tabular", "neuralprophet":"neuralprophet"}
            result={}
            for name in request["models"]:
                try:
                    importlib.import_module(modules[name])
                    result[name]={"import_ok":True,"version":providers.environment()["packages"][providers.PACKAGES[name]]}
                except Exception as error:
                    result[name]={"import_ok":False,"error":str(error)}
        else:
            if request["settings"].get("device","cpu") == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"]=""
            os.environ["HF_HUB_OFFLINE"]="1"
            data=np.load(request["input"])
            prediction=providers.predict(request["model"],data["y"],data["times"],request["horizon"],request["step_ns"],
                       request["settings"],request["output"],seed=request["seed"],lags=request["lags"],
                       season_length=request["season_length"],reuse=request.get("reuse"))
            result={"prediction":prediction.tolist()}
        response.write_text(json.dumps({"ok":True,"result":result}),encoding="utf-8")
    except BaseException as error:
        traceback.print_exc()
        response.write_text(json.dumps({"ok":False,"error":str(error),"traceback":traceback.format_exc()}),encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
