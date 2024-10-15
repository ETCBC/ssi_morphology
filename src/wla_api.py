import json
import requests
from urllib3.util import Retry

from config import wla_url


class WordLevelCorrectnessChecker:
    def __init__(self, wla_url, language, version):
        self.url = wla_url
        self.language = language #"syriac"
        self.version = version #"SSI"
        self.request = "analyse"
        self.mode = "text"

    def make_payload(self, prediction):
        """prediction: list[str] , contains predicted words"""
        payload = json.dumps(
                        {
                        "language": self.language,
                        "version": self.version,
                        "request": self.request,
                        "mode": self.mode,
                        "data": prediction
                        }
                            )
        return payload

    def make_api_request(self, payload):
        try:
            retry = Retry(
                total=3,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                )
            adapter = requests.adapters.HTTPAdapter(max_retries=retry)
            session = requests.Session()
            session.mount('https://', adapter)
            r = session.post(self.url, data=payload, timeout=0.001)

        except Exception as e:
            print(e)
            return('API Error')
        return res

