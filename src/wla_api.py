import json
import requests
from requests.packages.urllib3.util.retry import Retry
from typing import List

from config import wla_url
from enums import APICodes


class JakobCaller:
    def __init__(self, wla_url, language, version, predictions):
        self.url = wla_url
        self.language = language
        self.version = version
        self.request = "analyse"
        self.mode = "text"
        self.predictions = predictions

    def make_payload(self):
        """prediction: list[str] , contains predicted words"""
        payload = json.dumps(
                        {
                        "language": self.language,
                        "version": self.version,
                        "request": self.request,
                        "mode": self.mode,
                        "data": self.predictions
                        }
                            )
        return payload

    def make_api_request(self, payload):
        try:
            retry_strategy = Retry(
                                total=3,
                                backoff_factor=2,
                                status_forcelist=[429, 500, 502, 503, 504]
                                   )
            
            adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
            http = requests.Session()
            http.mount('https://', adapter)
            response = http.post(self.url, data=payload)

        except Exception as e:
            print(e)
            return('API Error')
        return response
    

class GrammarCorrectnessChecker:
    """
    Approach:
       Make api call with list of predicted strings.
       The first one that has grammatically correct value back is returned.
       If none of the predicted values is correct, the first in the list is returned with 'E' (error) attached to it.
    """
    def __init__(self, predictions, api_response):
        self.predictions = predictions
        self.api_response = api_response

    def load_response(self):
        return json.loads(self.api_response.text)
    
    def check_grammatical_correctness(self, response_dict: dict) -> tuple:
        results = response_dict.get('result', [])
        if not results:
            return self.predictions[0], APICodes['NOTCHECKED'].name
        else:
            for graphical_unit_analysis, graphical_unit_prediction in zip(results, self.predictions):
                errors_in_graphical_unit = all(['error' in word_analysis for word_analysis in graphical_unit_analysis[0]])
                if not errors_in_graphical_unit:
                    return graphical_unit_prediction, APICodes['CORRECT'].name
            return self.predictions[0], APICodes['NOTCORRECT'].name


def check_predictions(wla_url: str, language: str, version: str, predictions: List[str]) -> str:
    jakob_caller = JakobCaller(wla_url, language, version, predictions)
    payload = jakob_caller.make_payload()
    response = jakob_caller.make_api_request(payload)

    if response.status_code != 200:
        return predictions[0], APICodes['NOTCHECKED'].name
    
    correctness_checker = GrammarCorrectnessChecker(predictions, response)
    loaded_response = correctness_checker.load_response()
    prediction, api_code = correctness_checker.check_grammatical_correctness(loaded_response)
    if api_code == APICodes['CORRECT'].name:
        return f'{prediction}'
    return f'{prediction}\t{api_code}'