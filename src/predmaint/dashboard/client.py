"""The dashboard's only data source: the RUL API over HTTP (read-only GET requests)."""

import httpx

from predmaint import config

TRANSPORT: httpx.BaseTransport | None = None  # tests plug in httpx.MockTransport here


class ApiError(RuntimeError):
    """The API could not be reached or returned an error; the message is shown to the user."""


def api_url() -> str:
    return config.api_base_url()


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = base_url or api_url()
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=TRANSPORT)

    def _get(self, path: str, **params) -> object:
        try:
            r = self.http.get(path, params={k: v for k, v in params.items() if v is not None})
        except httpx.TransportError as exc:
            raise ApiError(
                f"Cannot reach the API at {self.base_url} ({type(exc).__name__}). Is it running? "
                "Start it with: uv run uvicorn predmaint.serving.api:app"
            ) from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"GET {path} failed with {r.status_code}: {detail}")
        return r.json()

    def health(self) -> dict:
        try:
            return self._get("/health")
        except ApiError as exc:
            if "failed with 503" in str(exc):  # API up, but no model loaded
                return {"status": "unavailable", "detail": str(exc)}
            raise

    def units(self) -> list[dict]:
        return self._get("/units")

    def predictions(self, unit_id: int, limit: int = 1000) -> list[dict]:
        return self._get(f"/units/{unit_id}/predictions", limit=limit)

    def readings(self, unit_id: int, limit: int | None = None) -> list[dict]:
        return self._get(f"/units/{unit_id}/readings", limit=limit)

    def drift(self, last_n_cycles: int = 30) -> dict:
        return self._get("/monitoring/drift", last_n_cycles=last_n_cycles)

    def close(self) -> None:
        self.http.close()
