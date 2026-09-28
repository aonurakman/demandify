from pathlib import Path

import httpx

from demandify.providers.osm import OSMFetcher


class _StubAsyncClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self._responses.pop(0)
        response.request = httpx.Request("POST", url)
        return response

    async def aclose(self):
        return None


def test_fetch_osm_data_uses_standard_overpass_query(tmp_path):
    fetcher = OSMFetcher(timeout=180)
    fetcher.client = _StubAsyncClient(
        [httpx.Response(200, content=b"<osm version='0.6'/>")]
    )

    output = tmp_path / "map.osm"
    bbox = (4.2817, 52.0696, 4.33, 52.0874)

    import asyncio

    asyncio.run(fetcher.fetch_osm_data(bbox, output, max_retries=1))

    assert output.read_bytes() == b"<osm version='0.6'/>"
    assert len(fetcher.client.calls) == 1
    _url, kwargs = fetcher.client.calls[0]
    query = kwargs["data"]["data"]
    assert query == (
        '[out:xml][timeout:180][bbox:52.0696,4.2817,52.0874,4.33];'
        '(way["highway"];node(w););'
        "out body;"
    )


def test_fetch_osm_data_retries_406_with_raw_query_body(tmp_path):
    fetcher = OSMFetcher(timeout=180)
    fetcher.client = _StubAsyncClient(
        [
            httpx.Response(406, text="Not Acceptable"),
            httpx.Response(200, content=b"<osm version='0.6'/>"),
        ]
    )

    output = tmp_path / "map.osm"
    bbox = (4.2817, 52.0696, 4.33, 52.0874)

    import asyncio

    asyncio.run(fetcher.fetch_osm_data(bbox, output, max_retries=1))

    assert output.read_bytes() == b"<osm version='0.6'/>"
    assert len(fetcher.client.calls) == 2

    _first_url, first_kwargs = fetcher.client.calls[0]
    _second_url, second_kwargs = fetcher.client.calls[1]

    assert "data" in first_kwargs
    assert first_kwargs["data"]["data"].startswith("[out:xml][timeout:180][bbox:")

    assert second_kwargs["content"].startswith("[out:xml][timeout:180][bbox:")
    assert second_kwargs["headers"]["Content-Type"] == "text/plain; charset=utf-8"
