"""Bake CPU-only, non-Hugging-Face providers into the backend image."""
import subprocess

from di_validator import store
from di_validator.forecasting.catalog import interpreter
from di_validator.forecasting.runtime import setup

class BuildProgress(store.Progress):
    def __call__(self, fraction, message):
        print(message, flush=True)


for model in ["gluonts", "autogluon", "neuralprophet"]:
    print("Preparing " + model, flush=True)
    setup({"model": model}, BuildProgress())
    subprocess.run([str(interpreter(model)), "-c",
                    "import importlib.util; assert importlib.util.find_spec('huggingface_hub') is None; "
                    "assert importlib.util.find_spec('transformers') is None"], check=True)
