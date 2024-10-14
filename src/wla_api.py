import json
import requests

from config import wla_url


class WordLevelCorrectnessChecker:
    def __init__(self, prediction):
        self.prediction = prediction
        self.url = wla_url
        self.language = "syriac"
        self.version = "SSI",
        self.request = "analyse",
        self.mode = "text"

    def make_payload(self):
        payload = json.dumps(
                        {
                        "language": self.language,
                        "version": self.version,
                        "request": self.request,
                        "mode": self.mode,
                        "data": [self.prediction]
                        }
                            )
        return payload

    def make_api_request(self, payload):
        try:
            res = requests.post(self.url, data=payload)
        except:
            pass
        return res

