"""Async access to DI Validator from JupyterLite. No workstation Python required.

Use ``await di.readings(...)`` for a bounded native-data page, or iterate
``async for frame in di.iter_frames(...)`` for a larger analysis. Data is read-only.
"""
from urllib.parse import urlencode, quote


class DIClient:
    def __init__(self, base_url=None):
        if base_url is None:
            from js import location
            base_url = str(location.origin) + "/api/v1"
        self.base_url = base_url.rstrip("/")

    async def _get(self, path, **params):
        from pyodide.http import pyfetch
        params = {k: v for k, v in params.items() if v is not None}
        url = self.base_url + path
        if params:
            url += "?" + urlencode(params)
        response = await pyfetch(url)
        data = await response.json()
        if not response.ok:
            raise ValueError(data.get("detail", f"Request failed: {response.status}"))
        return data

    async def datasets(self):
        return await self._get("/datasets")

    async def assets(self, dataset_id, search=""):
        return await self._get("/assets", dataset_id=dataset_id, search=search, limit=10000)

    async def quality(self, dataset_id):
        return await self._get(f"/datasets/{quote(dataset_id, safe='')}/quality")

    async def labels(self):
        """Manual label revisions; registry evidence is available through registries()."""
        return await self._get("/labels")

    async def registries(self):
        return await self._get("/notebooks/registries")

    async def registry(self, registry_id):
        """Return the normalized, versioned PV registry as a DataFrame."""
        import pandas as pd
        result = await self._get(f"/notebooks/registries/{quote(registry_id, safe='')}")
        frame = pd.DataFrame(result["rows"], columns=result["columns"])
        frame.attrs.update(result["metadata"])
        return frame

    async def annotations(self, dataset_id):
        return await self._get("/annotations", dataset_id=dataset_id)

    async def experiments(self):
        return await self._get("/experiments")

    async def predictions(self, experiment_id, model=None):
        return await self._get(f"/experiments/{quote(experiment_id, safe='')}/predictions", model=model)

    async def readings(self, dataset_id, *, asset_ids=None, channels=None,
                       start=None, end=None, limit=10000, cursor=None):
        """Fetch unchanged native readings (maximum 100,000 rows per page).

        Hourly bounds are ISO timestamps (naive times use the dataset timezone).
        Recording bounds are seconds from the recording origin. End is inclusive.
        A next_cursor resumes the same selection without losing equal timestamps.
        """
        return await self._get(f"/notebooks/data/{quote(dataset_id, safe='')}",
                               asset_ids=",".join(asset_ids) if asset_ids else None,
                               channels=",".join(channels) if channels else None,
                               start=start, end=end, limit=limit, cursor=cursor)

    @staticmethod
    def to_frame(page):
        import pandas as pd
        frame = pd.DataFrame(page["rows"], columns=page["columns"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601").astype("datetime64[ns, UTC]")
        numeric = {"value", "offset", "meter_coverage", "expected_meters", "contributing_meters"}
        numeric.update(c["name"] for c in page["metadata"]["channels"])
        for name in numeric.intersection(frame.columns):
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        frame.attrs.update(page["metadata"])
        return frame

    async def iter_frames(self, dataset_id, **selection):
        """Yield pages; consume each frame without concatenating whole recordings."""
        selection = dict(selection)
        while True:
            page = await self.readings(dataset_id, **selection)
            if page["rows"]:
                yield self.to_frame(page)
            if not page["next_cursor"]:
                break
            selection["cursor"] = page["next_cursor"]
