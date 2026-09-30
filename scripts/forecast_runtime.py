"""Install or verify isolated forecasting providers: uv run python -m scripts.forecast_runtime gluonts."""
import argparse
from di_validator.forecasting.runtime import setup

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("model")
    args=parser.parse_args()
    print(setup({"model":args.model}))
