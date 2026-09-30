FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY di_validator ./di_validator
COPY scripts ./scripts
COPY reference ./reference
COPY notebooks ./notebooks
COPY fault-distance/data/data_test ./fault-distance/data/data_test
COPY fault-distance/fault_distance.ipynb ./fault-distance/fault_distance.ipynb
COPY runtime/jupyterlite ./runtime/jupyterlite
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev && ln -s /data/sources /app/data
ENV PATH="/app/.venv/bin:$PATH" DI_HOSTED=1 DI_WORKSPACE=/data/runtime DI_FORECAST_RUNTIMES=/opt/forecast-runtimes
ENV DI_ALLOWED_HOSTS="*.run.app,localhost,127.0.0.1" DI_ALLOWED_ORIGINS="di-validator.n2wm787gnp.chatgpt.site"
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/matplotlib
RUN --mount=type=cache,target=/root/.cache/uv DI_WORKSPACE=/tmp/di-setup UV_TORCH_BACKEND=cpu python -m scripts.install_cloud_forecasts
EXPOSE 8080
CMD ["python", "-m", "di_validator.hosted"]
