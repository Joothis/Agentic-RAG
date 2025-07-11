from typing import Dict, Optional
import requests
import json
from langchain.tools import BaseTool

class APITool(BaseTool):
    name: str = "api_call"
    description: str = "Make HTTP requests to external APIs"
    base_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    api_key: Optional[str] = None
    session: Optional[requests.Session] = None

    def __init__(self, base_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None, api_key: Optional[str] = None):
        super().__init__()
        self.base_url = base_url
        self.headers = headers or {}
        self.api_key = api_key
        self.session = requests.Session()
        
        if headers:
            self.session.headers.update(headers)
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def _run(self, params: str) -> str:
        """Make an API call with the given parameters
        
        params should be a JSON string with the following keys:
        "method": "GET", "POST", "PUT", or "DELETE"
        "endpoint": the API endpoint
        "body": optional JSON object for the request body
        """
        try:
            # Parse parameters
            params_dict = json.loads(params)
            method = params_dict.get("method", "GET").upper()
            endpoint = params_dict.get("endpoint")
            body = params_dict.get("body")
            
            if not endpoint:
                return "Invalid parameters. 'endpoint' is required."
            
            # Construct URL
            url = f"{self.base_url}/{endpoint}" if self.base_url else endpoint
            
            # Make request
            response = self.session.request(
                method=method,
                url=url,
                json=body
            )
            
            # Return response
            return response.text
            
        except json.JSONDecodeError:
            return "Invalid JSON in parameters."
        except Exception as e:
            return f"Error making API call: {str(e)}"

    async def _arun(self, params: str) -> str:
        """Async implementation of run"""
        raise NotImplementedError("APITool does not support async")
