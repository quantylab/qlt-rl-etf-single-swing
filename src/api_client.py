import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests


DEFAULT_REQ_INTERVAL = 1.0
DEFAULT_BASE_URL = "https://api.quantylab.com"


def _today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def _past_yyyymmdd(days: int, base_date: Optional[str] = None) -> str:
    if base_date is None:
        base = datetime.now()
    else:
        base = datetime.strptime(base_date, "%Y%m%d")
    return (base - timedelta(days=days)).strftime("%Y%m%d")


class QuantylabRestClient:
    def __init__(self, token: str, base_url: str = DEFAULT_BASE_URL):
        if not token:
            raise ValueError("Quantylab API token is required.")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.last_request_time = 0.0

    def _throttle(self) -> None:
        interval = time.time() - self.last_request_time
        if interval < DEFAULT_REQ_INTERVAL:
            time.sleep(DEFAULT_REQ_INTERVAL - interval)
        self.last_request_time = time.time()

    def _get_json(self, url: str) -> Dict[str, Any]:
        self._throttle()
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _fetch_all(self, path: str, data_key: str) -> List[Dict[str, Any]]:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        records: List[Dict[str, Any]] = []
        while True:
            payload = self._get_json(url)
            records.extend(payload.get(data_key, []))
            next_path = payload.get("next")
            if not next_path:
                break
            url = next_path if next_path.startswith("http") else f"{self.base_url}{next_path}"
        return records

    def get_feature_vectors(
        self,
        code: str,
        version: str = "1",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        end_date = end_date or _today_yyyymmdd()
        start_date = start_date or _past_yyyymmdd((n or 20) + 7, base_date=end_date)
        path = (
            f"/feature-vectors?code={code}&version={version}"
            f"&start_date={start_date}&end_date={end_date}"
        )
        return self._fetch_all(path, data_key="data")

    def get_stock_market_candles(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        end_date = end_date or _today_yyyymmdd()
        start_date = start_date or _past_yyyymmdd(30, base_date=end_date)
        path = (
            f"/stock-market-candles?code={code}"
            f"&start_date={start_date}&end_date={end_date}"
        )
        return self._fetch_all(path, data_key="results")

    def get_stock_candles(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        end_date = end_date or _today_yyyymmdd()
        start_date = start_date or _past_yyyymmdd(30, base_date=end_date)
        path = (
            f"/stock-candles?code={code}"
            f"&start_date={start_date}&end_date={end_date}"
        )
        return self._fetch_all(path, data_key="data")
